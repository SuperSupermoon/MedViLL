import os
import json
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from fuzzywuzzy import fuzz
from datetime import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import AdamW
from transformers import BertTokenizer
from transformers import BertConfig, AutoConfig
from cnn_bert import CNN_BERT
import _pickle as cPickle
from utils import get_transforms, set_seed, truncate_txt, _create_entry, compute_score_with_logits


def _load_dataset(args, dataroot, name, img_id2val, label2ans):
    """Load entries
    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'"""
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])   

    entries = []
    for sample, answer in zip(samples, answers):
        # utils_for_vqa_test.assert_eq(sample['qid'], answer['qid'])
        # utils_for_vqa_test.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        if args.vqa_rad == 'all':
            entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == 'chest':
            if sample['image_organ'] in {'CHEST', ' CHEST', 'CHEST '}: entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == 'head':
            if sample['image_organ'] in {'HEAD', ' HEAD', 'HEAD '}: entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == 'abd':
            if sample['image_organ'] in {'ABD', ' ABD', 'ABD '}: entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries

def pre_processing(tokenizer, sentence): # 이것 사용함
    sentence = sentence.lower()
    if "? -yes/no" in sentence:
        sentence = sentence.replace("? -yes/no", "")
    if "? -open" in sentence:
        sentence = sentence.replace("? -open", "")
    if "? - open" in sentence:
        sentence = sentence.replace("? - open", "")
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
    token =  tokenizer(sentence)
    return token


class CXR_VQA_Dataset(Dataset):
    def __init__(self, src_file, tokenizer, transforms, args, data_set='train'):
        self.data_set = data_set
        ans2label_path = os.path.join(src_file, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(src_file, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.img_id2idx = json.load(open(os.path.join(src_file, 'imgid2idx.json')))
        self.entries = _load_dataset(args, src_file, self.data_set, self.img_id2idx, self.label2ans)
        self.args = args
        self.seq_len = args.seq_len
        self.transforms = transforms
        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        if args.bert_model == "bert-base-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        else:  # BERT-base, small, tiny
            self.BertTokenizer = BertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        question = self.entries[idx]['question']
        tokens = pre_processing(self.tokenizer, question)
        img_path = os.path.join(self.args.img_path ,self.entries[idx]['image_name'])

        if (self.entries[idx]['answer']['labels'] == []) or (self.entries[idx]['answer']['labels'] == None):
            self.entries[idx]['answer']['labels'] = None
            self.entries[idx]['answer']['scores'] = None            
        else:
            labels = np.array(self.entries[idx]['answer']['labels'])
            scores = np.array(self.entries[idx]['answer']['scores'], dtype=np.float32)     
            labels = torch.from_numpy(labels)
            scores = torch.from_numpy(scores)
            self.entries[idx]['answer']['labels'] = labels
            self.entries[idx]['answer']['scores'] = scores

        labels = self.entries[idx]['answer']['labels']
        scores = self.entries[idx]['answer']['scores']

        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.entries[idx]['answer_type'] in {'OPEN', 'OPEN ', ' OPEN'}:
            ans_type = torch.tensor([1])
        else:
            ans_type = torch.tensor([0]) # closed type

        sample = self.data_processing(tokens, img_path)   # tokens:['i', 'go', ..], str(이미지경로)
        return target, sample, ans_type

    def data_processing(self, tokens, img_path):
        #tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token  원래 str(text)를 받아서 여기서 token으로 만들어줬었는데 지금은 밖에서 token으로 만들고 가져온다
        truncate_txt(tokens, self.seq_len)

        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                            for w in tokens]  # [178, 8756, 1126, 12075]

        input_ids = [self.vocab_stoi["[CLS]"]] + encoded_sentence + [self.vocab_stoi["[SEP]"]]
        attn_masks = [1] * len(input_ids)

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 2)]    # self.seq_len: 253 인데 2를 더한건 맨 앞 cls와 맨 뒤 sep 때문. 결론적으로 패딩을 붙여서 255길이의 인풋을 만들거다

        input_ids.extend(padding)   # len(): 255
        attn_masks.extend(padding)  # len(): 255  앞은 쭉 1, 뒤는 쭉 0
        segment = [1 for _ in range(self.seq_len + 2)]   # len(): 255   # 여기선 segment 나눌 게 없음. BERT가 text만 받으니까. 그래서 다 1로 줌 

        input_ids = torch.tensor(input_ids)
        attn_masks = torch.tensor(attn_masks)
        segment = torch.tensor(segment)

        image = Image.open(img_path)
        image = transforms.Grayscale(num_output_channels=3)(image)
        image = self.transforms(image)

        return input_ids, attn_masks, segment, image

def train(args, train_dataset, eval_dataset, model, tokenizer):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(int(args.epochs)):
        train_losses = []
        train_data_iter = tqdm(enumerate(train_dataset), desc=f'EP_:{epoch}', total=len(train_dataset), bar_format='{l_bar}{r_bar}')
        for step, (labels, batch, ans_type) in train_data_iter:
            optimizer.zero_grad()
            labels = labels.to(args.device)
            input_txt = batch[0].to(args.device)    
            attn_mask = batch[1].to(args.device)
            segment = batch[2].to(args.device)
            input_img = batch[3].to(args.device)
            logits = model(input_txt, attn_mask, segment, input_img)   

            loss = criterion(logits, labels)   # # (B, num_class), (B, num_class)
            
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print(f'train loss: {round(loss.item(), 3)} ({round(np.mean(train_losses), 3)})')
        
        total_acc, closed_acc, open_acc = eval(args, eval_dataset, model, tokenizer)
        print({"avg_loss": np.mean(train_losses),
            "total_acc": total_acc,
            "closed_acc": closed_acc,
            "open_acc": open_acc})

    # save_path_per_ep = os.path.join(args.output_path, str(epoch))
    # os.makedirs(save_path_per_ep, exist_ok=True)
    # model.save_pretrained(save_path_per_ep)
    # print(f'EP: {epoch} Model saved on {save_path_per_ep}')    


def eval(args, eval_dataset, model, tokenizer):
    total_acc_list = []
    closed_correct_sum = 0
    open_correct_sum = 0
    total_closed_sum = 0
    total_open_sum = 0

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad(): 
        eval_losses = []
        eval_data_iter = tqdm(enumerate(eval_dataset), desc='EVAL: ', total=len(eval_dataset), bar_format='{l_bar}{r_bar}')

        for step, (labels, batch, ans_type) in eval_data_iter:    # ans_type=0: closed, ans_type=1:open
            labels = labels.to(args.device)
            input_txt = batch[0].to(args.device)    
            attn_mask = batch[1].to(args.device)
            segment = batch[2].to(args.device)
            input_img = batch[3].to(args.device)
            logits = model(input_txt, attn_mask, segment, input_img)   
            loss = criterion(logits, labels)   # (B, num_class), (B, num_class)
            eval_losses.append(loss.item())

            print(f'eval loss: {round(loss.item(), 3)} ({round(np.mean(eval_losses), 3)})')

            batch_score = compute_score_with_logits(logits, labels).sum(dim=1)   # (B, num_class) smu(dim=1)-> (B)
            batch_acc = batch_score.sum() / logits.size(0)
            total_acc_list.append(batch_acc.item())

            closed_score, open_score = [], []
            for i in range(len(ans_type)):
                if ans_type[i] == 0:  # closed
                    closed_score.append(batch_score[i].item())
                else:  # open
                    open_score.append(batch_score[i].item())            

            closed_correct = sum(closed_score)
            open_correct = sum(open_score)

            closed_num = len(closed_score)
            open_num = len(open_score)

            closed_correct_sum += closed_correct
            open_correct_sum += open_correct

            total_closed_sum += closed_num
            total_open_sum += open_num

        total_acc = sum(total_acc_list) / len(eval_dataset)  
        closed_acc = closed_correct_sum / total_closed_sum
        open_acc = open_correct_sum / total_open_sum
    return total_acc, closed_acc, open_acc

def main(args):
    set_seed(args.seed)
    cuda_condition = torch.cuda.is_available() and args.with_cuda
    args.device = torch.device("cuda" if cuda_condition else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(f'Device: {args.device}, n_gpu: {args.n_gpu}')

    if args.bert_model == "bert-base-scratch":
        config = BertConfig.from_pretrained("bert-base-uncased")
    else:
        config = BertConfig.from_pretrained(args.bert_model)  # bert-base, small, tiny.

    transforms = get_transforms(args)   # img_size: 512
    if args.bert_model == 'bert-base-scratch':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True).tokenize
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize

    if os.path.isfile(os.path.join(args.load_pretrained_model, 'pytorch_model.bin')):
        config = AutoConfig.from_pretrained(args.load_pretrained_model)
        model_state_dict = torch.load(os.path.join(args.load_pretrained_model, 'pytorch_model.bin'))
        model = CNN_BERT.from_pretrained(args.load_pretrained_model, state_dict=model_state_dict, config=config, args=args).to(args.device)
    else:
        model = CNN_BERT(config, args).to(args.device)

    if args.with_cuda and args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=args.cuda_devices)

    train_dataset = CXR_VQA_Dataset(args.src_file, tokenizer, transforms, args, 'train')
    test_dataset = CXR_VQA_Dataset(args.src_file, tokenizer, transforms, args, 'test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True)
    train(args, train_dataloader, test_dataloader, model, tokenizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa_rad", default="all", type=str,choices=["all", "chest", "head", "abd"])
    parser.add_argument("--num_class", default=458)
    parser.add_argument("--do_train", type=bool, default=True, help="Train & Evaluate")

    parser.add_argument("--src_file", type=str, default='/home/data_storage/mimic-cxr/dataset/data_RAD')
    parser.add_argument("--img_path", type=str, default='/home/data_storage/mimic-cxr/dataset/vqa_image/vqa_512_3ch')
    parser.add_argument("--train_dataset", type=str,                           
                        default='/home/data_storage/mimic-cxr/dataset/data_RAD/trainet.json',
                        help="train dataset for training")


    output_path = 'output/' + str(datetime.now())
    os.makedirs(output_path, exist_ok=True)

    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--epochs", type=int, default=10, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=16, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--hidden_size", type=int, default=768, choices=[768, 512, 128])
    parser.add_argument("--embedding_size", type=int, default=768, choices=[768, 512, 128])


    parser.add_argument("--load_pretrained_model", type=str, default='/home/edlab/jhmoon/mimic_mv_real/mimic-cxr/pre-train/base_PAR_36,128', choices=['','output/all/35','output/chest/45'])
                                 

    parser.add_argument("--bert_model", type=str, default="bert-base-scratch",
                        choices=["albert-base-v2",
                                 "bert-base-uncased",
                                 "google/bert_uncased_L-4_H-512_A-8",  # BERT-Small
                                 "google/bert_uncased_L-2_H-128_A-2",  # BERT-Tiny
                                 "emilyalsentzer/Bio_ClinicalBERT",  # Clinical-BERT
                                 "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",  # BlueBERT
                                 "bert-small-scratch",  # BERT-Small-scratch
                                 "bert-base-scratch", ])

    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000, 28996])  # 28996 clinical bert
    parser.add_argument("--img_postion", default=True, help='img_postion use!')
    parser.add_argument("--seq_len", type=int, default=253, help="maximum sequence len", choices=[253, 460])  # 253
    parser.add_argument("--max_seq_len", type=int, default=512, help="total sequence len")
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_channel", type=int, default=1, choices=[1, 3])
    parser.add_argument("--img_size", type=int, default=224, choices=[224, 512])  # TODO: change helper.py, resize(224)

    parser.add_argument("--lr", type=float, default=3e-5)
    
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    main(args)