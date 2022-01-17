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

from helper import get_transforms
from cnn_bert import CNN_BERT

import _pickle as cPickle

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()   # pop()은 리스트의 맨 마지막 요소를 돌려주고 그 요소는 삭제

def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('image_name')    # pop은 리스트에서 해당 원소를 삭제해주는 함수
        answer.pop('qid')

    entry = {
        'qid' : data['qid'],
        'image_name'   : data['image_name'],
        'image'        : img,
        'question'     : data['question'],
        'answer'       : answer,
        'answer_type'  : data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type'  : data['phrase_type'],
        'image_organ'  : data['image_organ']}
    return entry

def _load_dataset(args, dataroot, name, img_id2val, label2ans):   # 이거 사용함
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """

    #img_id2val  전체 임지 로드 {key : 이미지 이름, value: index 매핑}

    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])   
    # print("np.unique(answers)",np.unique(answers['scores']))

    # utils_for_vqa_test.assert_eq(len(samples), len(answers))
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
    def __init__(self, src_file, tokenizer, transforms, args, is_train=True):
        if is_train == True:
            self.data_set = 'train'
        else:
            self.data_set = 'test'
        ans2label_path = os.path.join(src_file, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(src_file, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))   # key: value(0부터 쭉~)
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))   # key만 있음
        self.num_ans_candidates = len(self.ans2label) #458개 있음
        self.img_id2idx = json.load(open(os.path.join(src_file, 'imgid2idx.json'))) # 전체 이미지 로드. dict. len():315
        self.entries = _load_dataset(args, src_file, self.data_set, self.img_id2idx, self.label2ans) # dict을 원소로 갖는 리스트. train이라면 len(): 3064. test라면 len(): 451
        
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


        #labels = np.array(answer['labels'])
        #scores = np.array(answer['scores'], dtype=np.float32)     
        #labels = torch.from_numpy(labels)
        #scores = torch.from_numpy(scores)

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

        if self.args.img_channel == 3:
            image = Image.open(img_path)
        elif self.args.img_channel == 1:
            image = Image.open(img_path)
            image = transforms.Grayscale(num_output_channels=3)(image)

        image = self.transforms(image)

        return input_ids, attn_masks, segment, image




def train(args, train_dataset, eval_dataset, model, tokenizer):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()    # nn.CrossEntropyLoss() 써도 되는데 똑같은 조건을 위해 nn.BCEWithLogitsLoss()을 쓰자
    for epoch in range(int(args.epochs)):
        train_losses = []

        train_data_iter = tqdm(enumerate(train_dataset),
                               desc=f'EP_:{epoch}',
                               total=len(train_dataset),
                               bar_format='{l_bar}{r_bar}')
    
        for step, (labels, batch, ans_type) in train_data_iter:
            model.train()
            labels = labels.to(args.device)
            input_txt = batch[0].to(args.device)    
            attn_mask = batch[1].to(args.device)
            segment = batch[2].to(args.device)
            input_img = batch[3].to(args.device)
            logits = model(input_txt, attn_mask, segment, input_img)   

            loss = criterion(logits, labels)   # # (B, num_class), (B, num_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            

            print(f'train loss: {round(loss.item(), 3)} ({round(np.mean(train_losses), 3)})')
            
        

        save_path_per_ep = os.path.join(args.output_path, str(epoch))
        os.makedirs(save_path_per_ep, exist_ok=True)

        if args.n_gpu > 1:
            model.module.save_pretrained(save_path_per_ep)
            print(f'Multi_EP: {epoch} Model saved on {save_path_per_ep}')
        else:
            model.save_pretrained(save_path_per_ep)    # 이렇게 하면 config.json파일이랑 pytorch_model.bin이 만들어짐
            print(f'Single_EP: {epoch} Model saved on {save_path_per_ep}')

        
        total_acc, closed_acc, open_acc = eval(args, eval_dataset, model, tokenizer)
        print({"avg_loss": np.mean(train_losses),
                "total_acc": total_acc,
                "closed_acc": closed_acc,
                "open_acc": open_acc}, step=epoch)



def eval(args, eval_dataset, model, tokenizer):
    total_acc_list = []
    closed_correct_sum = 0
    open_correct_sum = 0
    total_closed_sum = 0
    total_open_sum = 0

    criterion = nn.BCEWithLogitsLoss()    # nn.CrossEntropyLoss() 써도 되는데 똑같은 조건을 위해 nn.BCEWithLogitsLoss()을 쓰자
    model.eval()
    with torch.no_grad(): 
        eval_losses = []
        eval_data_iter = tqdm(enumerate(eval_dataset),
                               desc='EVAL: ',
                               total=len(eval_dataset),
                               bar_format='{l_bar}{r_bar}')

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

            closed_score = []
            open_score = []
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

            total_closed_sum +=  closed_num
            total_open_sum += open_num

        total_acc = sum(total_acc_list) / len(eval_dataset)  
        closed_acc = closed_correct_sum / total_closed_sum
        open_acc = open_correct_sum / total_open_sum
        
    return total_acc, closed_acc, open_acc

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores   # (B, 458)



def main(args):

    set_seed(args.seed)

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    args.device = torch.device("cuda" if cuda_condition else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(f'Device: {args.device}, n_gpu: {args.n_gpu}')

    if args.bert_model == "bert-base-scratch":
        config = BertConfig.from_pretrained("bert-base-uncased")     # 타입이 클래스임
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

    if args.do_train:
        print("Load Train dataset", args.train_dataset)            ### 여기 중요!
        train_dataset = CXR_VQA_Dataset(args.src_file, tokenizer, transforms, args, is_train=True)   ### 여기 중요! CXR_VQA_Dataset 정의하기
        test_dataset = CXR_VQA_Dataset(args.src_file, tokenizer, transforms, args, is_train=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True)
        train(args, train_dataloader, test_dataloader, model, tokenizer)

    else:
        pass   # 현재 train, eval, test로 나눠져 있지 않고 train, test만 있어서 따로 test할 셋이 없다. 








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa_rad", default="all", type=str,     # 사용됨. dataset에서.
                            choices=["all", "chest", "head", "abd"])
    parser.add_argument("--num_class", default=458)
    parser.add_argument("--do_train", type=bool, default=True, help="Train & Evaluate")


    parser.add_argument("--src_file", type=str, default='/home/mimic-cxr/dataset/data_RAD')
    parser.add_argument("--img_path", type=str, default='/home/mimic-cxr/dataset/vqa_image/vqa_512_3ch/')
    parser.add_argument("--train_dataset", type=str,                           
                        default='/home/mimic-cxr/dataset/data_RAD/trainet.json',
                        help="train dataset for training")


    output_path = 'output/' + str(datetime.now())
    os.makedirs(output_path, exist_ok=True)

    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--epochs", type=int, default=10, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=14, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--hidden_size", type=int, default=768, choices=[768, 512, 128])
    parser.add_argument("--embedding_size", type=int, default=768, choices=[768, 512, 128])


    parser.add_argument("--load_pretrained_model", type=str,
                        default='',
                        choices=['',
                                 'output/all/35',
                                 'output/chest/45'])
                                 

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