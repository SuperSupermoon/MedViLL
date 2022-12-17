import math
import torch
import torch.nn as nn
from models.image import ImageEncoder_cnn, Img_patch_embedding
from transformers import BertConfig, BertModel, BertPreTrainedModel

class ImageBertEmbeddings(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.img_embeddings = nn.Linear(2048, 768)
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(0.1)
        self.position_embeddings = embeddings.position_embeddings

    def forward(self, input_imgs, img_pos, token_type_ids):
        imgs_embeddings = self.img_embeddings(input_imgs)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        position_embeddings = self.position_embeddings(img_pos)
        embeddings = imgs_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MedViLLEncoder(BertPreTrainedModel):
    def __init__(self, model_config, args, configs):
        super().__init__(model_config)
        self.args = args
        self.configs = configs
        bert = BertModel(model_config)

        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(self.txt_embeddings)

        if configs['img_encoder'] == 'ViT':
            img_size = configs['img_size']
            patch_sz = 32 if img_size == 512 else 16
            self.img_encoder = Img_patch_embedding(image_size=img_size, patch_size=patch_sz, dim=2048)
        else:
            self.img_encoder = ImageEncoder_cnn(args, configs)
            for p in self.img_encoder.parameters():
                p.requires_grad = False
                
            # We train some part of the layer on the cnn model.
            for c in list(self.img_encoder.children())[5:]: 
                for p in c.parameters():
                    p.requires_grad = True

        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def get_extended_attn_mask(self, attn_mask):
        if attn_mask.dim() == 2:
            extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        elif attn_mask.dim() == 3:
            extended_attn_mask = attn_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)
        extended_attn_mask = (1.0 - extended_attn_mask) * - 10000.0
        return extended_attn_mask

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        extended_attn_mask = self.get_extended_attn_mask(attn_mask)
        if self.args.disturbing_mask:
            img_tok = (torch.LongTensor(input_txt.size(0), (self.configs['num_image_embeds'])).fill_(0).cuda())
            sep_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())
            cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())
            txt_cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(1).cuda())
            txt_cls_out = self.txt_embeddings(cls_tok, txt_cls_segment)
            img, img_pos = self.img_encoder(input_img)  # BxNx2048

            sep_out = self.txt_embeddings(sep_tok, sep_segment)
            cls_out = self.txt_embeddings(cls_tok, cls_segment)
            img_embed_out = self.img_embeddings(img, img_pos, img_tok)
            txt_embed_out = self.txt_embeddings(input_txt, segment)
            encoder_input = torch.cat([cls_out, img_embed_out, sep_out, txt_cls_out, txt_embed_out],1)  # TODO: Check B x (TXT + IMG) x HID
            encoded_layers = self.encoder(
                encoder_input, extended_attn_mask, output_hidden_states=False)
            cls_represent = encoded_layers[-1][:, 0] * encoded_layers[-1][:,self.configs['num_image_embeds'] + 2]  # element wisely multiplication.
            return encoded_layers[-1], cls_represent

        else:
            img_tok = (torch.LongTensor(input_txt.size(0), self.configs['num_image_embeds']).fill_(0).cuda())
            cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())
            cls_out = self.txt_embeddings(cls_tok, cls_segment)
            sep_out = self.txt_embeddings(sep_tok, cls_segment)
            img, position = self.img_encoder(input_img)
            img_embed_out = self.img_embeddings(img, position, img_tok)  # bsz, num_img_embeds, hsz
            txt_embed_out = self.txt_embeddings(input_txt, segment)  # bsz, seq_len, hsz. inputs: bsz, seq_len
            encoder_input = torch.cat([cls_out, img_embed_out, sep_out, txt_embed_out], 1)  # B x (TXT + IMG) x HID
            encoded_layers = self.encoder(encoder_input, extended_attn_mask, output_hidden_states=False, output_attentions=True)
            return encoded_layers[0], self.pooler(encoded_layers[0]), encoded_layers[1]

class MedViLL(BertPreTrainedModel):
    """
    Multimodal BERT
    : Masked Language Model + Image Text Matching
    """
    def __init__(self, model_config, args, configs):
        super().__init__(model_config)
        self.enc = MedViLLEncoder(model_config, args, configs)
        self.mlm = BertPreTrainingHeads(model_config, self.enc.txt_embeddings.word_embeddings.weight)
        self.itm = ImageTextMatching(configs['hidden_size'])

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        x_mlm, x_itm, _ = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)
        prediction_scores_masked, _ = self.mlm(x_mlm)
        predict_itm = self.itm(x_itm)
        return prediction_scores_masked, predict_itm

class ImageTextMatching(nn.Module):
    """
    2-class classification model : Aligned, Not aligned
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.linear(x)

# reproduce mlm referred VLP to impact generation task
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, model_config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[model_config.hidden_act] \
            if isinstance(model_config.hidden_act, str) else model_config.hidden_act
        hid_size = model_config.hidden_size
        self.dense = nn.Linear(model_config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, model_config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(model_config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self, model_config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            model_config, bert_model_embedding_weights)
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = None
        return prediction_scores, seq_relationship_score