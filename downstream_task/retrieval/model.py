"""
Model for retrieval task
"""
import os
import torch

from transformers import AutoConfig
from transformers import BertConfig, BertPreTrainedModel
from cxrbert_origin import CXRBERT

class CXRBertForRetrieval(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        if args.weight_load:
            config = AutoConfig.from_pretrained(args.load_pretrained_model)
            model_state_dict = torch.load(os.path.join(args.load_pretrained_model, 'pytorch_model.bin'))
            cxrbert = CXRBERT.from_pretrained(args.load_pretrained_model,
                                              state_dict=model_state_dict, config=config, args=args)
        else:
            config = BertConfig.from_pretrained('bert-base-uncased')
            cxrbert = CXRBERT(config, args)

        self.enc = cxrbert.enc
        self.itm = cxrbert.itm

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        _, cls, _ = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)
        result = self.itm(cls)
        return result
