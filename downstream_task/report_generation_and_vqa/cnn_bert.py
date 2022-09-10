"""
CNN + BERT
    1. CNN(resnet50), GAP
    2. BERT Encoder, CLS token
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import wandb
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from fuzzywuzzy import fuzz
from datetime import datetime


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers.optimization import AdamW
from transformers import BertTokenizer, AutoTokenizer
from transformers.modeling_auto import AutoConfig, AutoModel
from transformers.modeling_bert import BertConfig, BertModel, BertPreTrainedModel
from utils import get_transforms, set_seed, truncate_txt



class CNN_BERT_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(line) for line in open(data_path)]

        self.num_image_embeds = args.num_image_embeds
        self.seq_len = args.seq_len
        self.transforms = transforms

        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        if args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 28996

        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-small-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-base-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        else:  # BERT-base, small, tiny
            self.BertTokenizer = BertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, _, label, txt, img = self.data[idx].keys()
        origin_txt = self.data[idx][txt]
        img_path = self.data[idx][img]

        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token

        truncate_txt(tokenized_sentence, self.seq_len)

        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                            for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids = [self.vocab_stoi["[CLS]"]] + encoded_sentence + [self.vocab_stoi["[SEP]"]]

        attn_masks = [1] * len(input_ids)

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 2)]

        input_ids.extend(padding)
        attn_masks.extend(padding)

        segment = [1 for _ in range(self.seq_len + 2)]

        input_ids = torch.tensor(input_ids)
        attn_masks = torch.tensor(attn_masks)
        segment = torch.tensor(segment)

        if self.args.img_channel == 3:
            image = Image.open(os.path.join(self.data_dir, img_path))
        elif self.args.img_channel == 1:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

        image = self.transforms(image)

        return input_ids, attn_masks, segment, image

class IMG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = F.adaptive_avg_pool2d
        for p in self.model.parameters():
            p.requires_grad = False
        # only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True

    def forward(self, x):
        out = self.model(x)  # (B, 3, 512, 512) -> (B, 2048, 16, 16)
        # out = self.pool(out, (1, 1))  # 
        out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        return out

class TXT_Encoder(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

        if args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            bert = AutoModel.from_pretrained(args.bert_model)
        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            bert = AutoModel.from_pretrained(args.bert_model)
        elif args.bert_model == "bert-small-scratch":
            config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            bert = BertModel(config)
        elif args.bert_model == "bert-base-scratch":
            config = BertConfig.from_pretrained("bert-base-uncased")
            bert = BertModel(config)
        else:
            bert = BertModel.from_pretrained(args.bert_model)  # bert-base-uncased, small, tiny

        self.txt_embeddings = bert.embeddings

        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def get_extended_attn_mask(self, attn_mask):
        if attn_mask.dim() == 2:    # eg. (B, 255) -> (B, 1, 1, 255) 이렇게 만들어주고 BERT에 넣어주면 알아서 내부적으로 (B, 1, 255, 255)을 만듦 
            extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        elif attn_mask.dim() == 3:  # eg. (B, 255, 255) -> (B, 1, 255, 255)
            extended_attn_mask = attn_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)
        extended_attn_mask = (1.0 - extended_attn_mask) * - 100000000000

        return extended_attn_mask

    def forward(self, input_txt, attn_mask, segment):
        extended_attn_mask = self.get_extended_attn_mask(attn_mask)
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # bsz, seq_len, hsz. inputs: bsz, seq_len
        encoded_layers = self.encoder(txt_embed_out, extended_attn_mask, output_hidden_states=False)   # 맨 마지막 레이어의 output. 따라서 
        return self.pooler(encoded_layers[0])


class CNN_BERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        img_txt_hiddens = args.img_hidden_sz + args.hidden_size   # 이미지 벡터랑 text벡터(from BERT CLS)를 concat할 것이기 때문

        self.txt_enc = TXT_Encoder(config, args)
        self.img_enc = IMG_Encoder()
        self.linear = nn.Linear(img_txt_hiddens, args.num_class)

    def forward(self, input_txt, attn_mask, segment, input_img):
        txt_cls = self.txt_enc(input_txt, attn_mask, segment)
        img_cls = self.img_enc(input_img)

        cls = torch.cat([img_cls, txt_cls], 1)  # (B, 2816)

        return self.linear(cls)


def train(args, config, train_dataloader, eval_dataloader, model):
    pass


def test(args, config, eval_dataloader, model):
    pass



def main(args):
    wandb.init(config=args, project='CNN_BERT')
    set_seed(args.seed)

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    args.device = torch.device("cuda" if cuda_condition else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(f'Device: {args.device}, n_gpu: {args.n_gpu}')

    if args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
        config = AutoConfig.from_pretrained(args.bert_model)
    elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
        config = AutoConfig.from_pretrained(args.bert_model)
    elif args.bert_model == "bert-small-scratch":
        config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
    elif args.bert_model == "bert-base-scratch":
        config = BertConfig.from_pretrained("bert-base-uncased")
    else:
        config = BertConfig.from_pretrained(args.bert_model)  # bert-base, small, tiny

    transforms = get_transforms(args)

    if args.bert_model == 'bert-base-scratch':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True).tokenize
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize

    model = CNN_BERT(config, args).to(args.device)
    wandb.watch(model)

    if args.with_cuda and args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=args.cuda_devices)

    if args.do_train:
        print('Load Train dataset', args.train_dataset)
        print('Load Valid dataset', args.valid_dataset)
        train_dataset = CNN_BERT_Dataset(args.train_dataset, tokenizer, transforms, args)
        val_dataset = CNN_BERT_Dataset(args.valid_dataset, tokenizer, transforms, args)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=True)
        eval_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        train(args, config, train_dataloader, eval_dataloader, model)

    if args.do_test:
        test_dataset = CNN_BERT_Dataset(args.test_dataset, tokenizer, transforms, args)
        eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        test(args, config, eval_dataloader, model)
