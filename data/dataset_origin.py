"""
generate dataset
"""
import os
import json
import torch
import random
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
import torchvision.transforms as transforms

def get_transforms():
    return transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()

class CXRDataset(Dataset):
    def __init__(self, data_path, tokenizer, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        self.seq_len = args.seq_len
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len  # 512
        self.max_seq_len -= args.num_image_embeds  # 512 - #img_embeds
        self.total_len = self.seq_len + self.args.num_image_embeds + 3
        self._tril_matrix = torch.tril(torch.ones((self.total_len, self.total_len), dtype=torch.long))
        self.BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_stoi = self.BertTokenizer.vocab
        self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # MLM
        origin_txt, img_path, is_aligned, itm_prob = self.random_pair_sampling(idx)
        
        change_path = img_path.split('/')
        fixed_path = change_path[:-2]
        fixed_path = "/".join(fixed_path)
        static_path = change_path[-2:]
        static_path = "/".join(static_path)

        if fixed_path == '/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch':
            fixed_path = '/home/data_storage/mimic-cxr/dataset/image_preprocessing/re_512_3ch/'
            img_path = fixed_path + static_path
            
        if self.args.img_channel == 3:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")
        elif self.args.img_channel == 1:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

        image = get_transforms()(image)
        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token
        truncate_txt(tokenized_sentence, self.seq_len)

        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                            for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        if self.args.disturbing_mask:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = [-100] + txt_labels + [-100]
            txt_labels_i = [-100] * (self.args.num_image_embeds + 2)
        else:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = txt_labels + [-100]
            txt_labels_i = [-100] * (self.args.num_image_embeds + 2)

        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * (self.args.num_image_embeds + 2)

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
        label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]

        input_ids.extend(padding)
        attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        attn_masks = attn_masks_i + attn_masks_t  
        
        segment = [1 for _ in range(self.seq_len + 1)]  # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids_tensor = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)
        is_aligned = torch.tensor(is_aligned)

        attn_1d = torch.tensor(attn_masks)

        full_attn = torch.tensor((attn_masks_i + attn_masks_t),
                                 dtype=torch.long).unsqueeze(0).expand(self.total_len, self.total_len).clone()

        extended_attn_masks = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
        second_st, second_end = self.args.num_image_embeds + 2, self.args.num_image_embeds + 2 + len(input_ids)
        extended_attn_masks[:, :self.args.num_image_embeds + 2].fill_(1)
        extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end - second_st, :second_end - second_st])
        s2s_attn = extended_attn_masks

        mixed_lst = [full_attn, s2s_attn]

        if self.args.Mixed:
            assert (self.args.s2s_prob + self.args.bi_prob) == 1.0
            attn_masks_tensor = random.choices(mixed_lst, weights=[self.args.bi_prob, self.args.s2s_prob])[0]

        elif self.args.BAR_attn:
            extended_attn_masks[:self.args.num_image_embeds+2, :].fill_(1)
            attn_masks_tensor = extended_attn_masks

        elif self.args.disturbing_mask:
            baseline_attn = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
            baseline_attn[:self.args.num_image_embeds + 2, :self.args.num_image_embeds + 2].fill_(1)
            baseline_attn[self.args.num_image_embeds + 2:, self.args.num_image_embeds + 2:].fill_(1)
            attn_masks_tensor = baseline_attn

        else:
            if self.args.attn_1d:
                attn_masks_tensor = attn_1d  # '1d attention mask'

            else:
                attn_masks_tensor = full_attn  # 'Full attention mask'

        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        return cls_tok, input_ids_tensor, txt_labels, attn_masks_tensor, image, segment, is_aligned, sep_tok, itm_prob

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab_stoi["[MASK]"]
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab_len)

                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab_stoi["[MASK]"]
            
        return tokens, output_label

    def random_pair_sampling(self, idx):
        _, _, label, txt, img = self.data[idx].keys()  # id, txt, img

        d_label = self.data[idx][label]
        d_txt = self.data[idx][txt]
        d_img = self.data[idx][img]
        itm_prob = random.random()

        if itm_prob > 0.5:
            return d_txt, d_img, 1, itm_prob
        else:
            for itr in range(300):
                random_txt, random_label = self.get_random_line()
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_txt, d_img, 0, itm_prob
                    break
                else:
                    pass

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']
        label = self.data[rand_num]['label']
        return txt, label
