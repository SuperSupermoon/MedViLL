"""
generate dataset
"""
import os
import ast
import json
import torch
import random
import pandas as pd
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from torch.utils.data import Dataset
from transformers import BertModel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_transforms():
    return transforms.Compose(
        [transforms.RandomResizedCrop(512, scale=(0.8, 1.1), ratio=(3/4, 4/3)),   
         transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()

class CXRDataset(Dataset):
    def __init__(self, data_path, tokenizer, args, config):
        self.args = args
        self.config = config
        if data_path.split('.')[-1] == 'jsonl':    
            self.data_dir = os.path.dirname(data_path)
            self.data = [json.loads(l) for l in open(data_path)]
        else:
            assert data_path.endswith(".csv")
            self.data = pd.read_csv(data_path)       
        self.data_path = data_path     
        self.seq_len = config['seq_len']
        self.tokenizer = tokenizer
        self.max_seq_len = config['max_seq_len']  # 512
        self.max_seq_len -= config['num_image_embeds']  # 512 - #img_embeds
        self.total_len = self.seq_len + self.config['num_image_embeds'] + 3
        self._tril_matrix = torch.tril(torch.ones((self.total_len, self.total_len), dtype=torch.long))
        self.vocab_stoi = self.tokenizer.vocab
        self.vocab_len = len(self.vocab_stoi)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # MLM
        if self.data_path.endswith(".jsonl"):
            origin_txt, img_path, is_aligned = self.random_pair_sampling(idx)        
        elif self.data_path.endswith(".csv"):
            image_path_list = ast.literal_eval(self.data["image"][idx])
            img_path = random.choice(image_path_list)
            text = ast.literal_eval(self.data["text"][idx])
            if isinstance(text, list):
                origin_txt = random.choice(text)
            #(Mock value)
            is_aligned = 1

        image = Image.open(os.path.join(img_path)).convert("RGB")

        image = get_transforms()(image)
        
        tokenized_sentence = self.tokenizer(origin_txt)
        truncate_txt(tokenized_sentence, self.seq_len)

        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                            for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        if self.args.disturbing_mask:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = [-100] + txt_labels + [-100]
            txt_labels_i = [-100] * (self.config['num_image_embeds'] + 2)
        else:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = txt_labels + [-100]
            txt_labels_i = [-100] * (self.config['num_image_embeds'] + 2)

        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * (self.config['num_image_embeds'] + 2)

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
        label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]

        input_ids.extend(padding)
        attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        # attn_masks = attn_masks_i + attn_masks_t  
        
        segment = [1 for _ in range(self.seq_len + 1)]  # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids_tensor = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)
        is_aligned = torch.tensor(is_aligned)

        full_attn = torch.tensor((attn_masks_i + attn_masks_t),
                                 dtype=torch.long).unsqueeze(0).expand(self.total_len, self.total_len).clone()

        extended_attn_masks = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
        second_st, second_end = self.config['num_image_embeds'] + 2, self.config['num_image_embeds'] + 2 + len(input_ids)
        extended_attn_masks[:, :self.config['num_image_embeds'] + 2].fill_(1)
        extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end - second_st, :second_end - second_st])
        s2s_attn = extended_attn_masks

        mixed_lst = [full_attn, s2s_attn]

        if self.args.Mixed:
            assert (self.args.s2s_prob + self.args.bi_prob) == 1.0
            attn_masks_tensor = random.choices(mixed_lst, weights=[self.args.bi_prob, self.args.s2s_prob])[0]

        elif self.args.BAR_attn:
            extended_attn_masks[:self.config['num_image_embeds']+2, :].fill_(1)
            attn_masks_tensor = extended_attn_masks

        elif self.args.disturbing_mask:
            baseline_attn = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
            baseline_attn[:self.config['num_image_embeds'] + 2, :self.config['num_image_embeds'] + 2].fill_(1)
            baseline_attn[self.config['num_image_embeds'] + 2:, self.config['num_image_embeds'] + 2:].fill_(1)
            attn_masks_tensor = baseline_attn

        else:
            attn_masks_tensor = full_attn  # 'Full attention mask'

        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        return cls_tok, input_ids_tensor, txt_labels, attn_masks_tensor, image, segment, is_aligned, sep_tok

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
            return d_txt, d_img, 1
        else:
            for itr in range(300):
                random_txt, random_label = self.get_random_line()
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_txt, d_img, 0
                    break
                else:
                    pass

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']
        label = self.data[rand_num]['label']
        return txt, label

def create_dataset(tokenizer, config, args):
    train_dataset = CXRDataset(
                data_path=config['train_dataset'],
                args=args,
                tokenizer = tokenizer,
                config=config)
    valid_dataset = CXRDataset(
                data_path=config['valid_dataset'],
                args=args,
                tokenizer = tokenizer,
                config=config)
    test_dataset = CXRDataset(
                data_path=config['test_dataset'],
                args=args,
                tokenizer = tokenizer,
                config=config)
    return [train_dataset, valid_dataset, test_dataset]

def create_sampler(datasets, shuffles, num_gpus, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_gpus, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, is_trains, num_workers=0):
    loaders = []
    for dataset,sampler,bs,is_train in zip(datasets,samplers,batch_size,is_trains):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    