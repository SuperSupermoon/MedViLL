"""BERT finetuning runner."""
import os
import copy
import logging
import glob
import json
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle
import torch
import torch.nn as nn
import torchvision
import csv
from transformers import BertTokenizer
from pytorch_pretrained_bert.model import BertForSeq2SeqDecoder
import wandb
# import sc.seq2seq_loader_itm as seq2seq_loader
from data_loader import Preprocess4Seq2seqDecoder
from bleu import language_eval_bleu
from data_parallel import DataParallelImbalance
from collections import defaultdict
import pickle
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(" # PID :", os.getpid())

def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_postion", type=str, default=True, choices=[True | False])
    parser.add_argument("--img_encoding", type=str, default='fully_use_cnn', choices=['random_sample', 'fully_use_cnn'])
    parser.add_argument('--img_hidden_sz', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument("--global_rank",
                    type=int,
                    default=-1,
                    help="global_rank for distributed training on gpus")
    parser.add_argument("--random_bootstrap_testnum",
                    type=int,
                    default=30,
                    help="global_rank for distributed training on gpus")
    parser.add_argument("--eval_model", default='pretrained_', type=str, help="designate your test model to extract your label from gen report.")

    parser.add_argument('--wandb', action='store_true', default = False,
                        help="Whether to use wandb logging")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.") # model load
    # parser.add_argument("--model_recover_path", default='/home/mimic-cxr/downstream_model/report_generation/base_mimic_par_256_128/model.50.bin', type=str,
    #                     help="The file of fine-tuned pretraining model.") # model load

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased")
    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help="max position embeddings")
                    
    # For decoding
    parser.add_argument('--fp16', action='store_true', default= False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true', default = False,
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=60,
                        help="Batch size for decoding.") #160
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")

    parser.add_argument('--sampling_case', type=int, default=-1,
                        help="maximum length of decode samplig number")

    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    # 
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--max_txt_length', type=int, default=128,
                        help="maximum length of target sequence")

    parser.add_argument('--dataset', default='cxr', type=str)
    parser.add_argument('--len_vis_input', type=int, default=256)
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--file_valid_jpgs', default='', type=str)


    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # setting gpu number
    args.config_path = args.model_recover_path.split('/')[:-1]
    args.config_path = ('/').join(args.config_path)+'/config.json'

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    args.max_seq_length = args.max_txt_length + args.len_vis_input + 3 # +3 for 2x[SEP] and [CLS]
    # tokenizer.max_len = args.max_seq_length

    bi_uni_pipeline = []
    bi_uni_pipeline.append(Preprocess4Seq2seqDecoder(tokenizer, args.max_seq_length,
        max_txt_length=args.max_txt_length, new_segment_ids=args.new_segment_ids,
        mode='s2s', len_vis_input=args.len_vis_input))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2

    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])
    
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))

    max_a, max_b, max_c, max_d = [], [], [], []
    results_dict = defaultdict(list)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        for bootstrap in range(1,args.random_bootstrap_testnum+1):
            logger.info("***** Recover model: %s *****", args.model_recover_path)
            model_recover = torch.load(args.model_recover_path)

            for key in list(model_recover.keys()):
                model_recover[key.replace('txt_embeddings', 'bert.txt_embeddings'). replace('img_embeddings', 'bert.img_embeddings'). replace('img_encoder.model', 'bert.img_encoder.model'). replace('encoder.layer', 'bert.encoder.layer'). replace('pooler', 'bert.pooler')] = model_recover.pop(key)

            for key in list(model_recover.keys()):
                model_recover[key.replace('bert.img_embeddings.bert.img_embeddings','bert.img_embeddings.img_embeddings')] = model_recover.pop(key)

            model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
                max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
                state_dict=model_recover, args=args, num_labels=cls_num_labels,
                type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id,
                search_beam_size=args.beam_size, length_penalty=args.length_penalty,
                eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
                forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len,
                len_vis_input=args.len_vis_input)

            model.load_state_dict(model_recover, strict=False)
            del model_recover

            if args.wandb and utils.is_main_process():
                wandb.init(config=args, project='Medvill', entity='mimic-cxr')

            model.to(device)

            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            if args.wandb and utils.is_main_process():
                wandb.watch(model)

            torch.cuda.empty_cache()

            eval_lst = []
            if 'openi' in re.split(r'[ _/]', args.model_recover_path):
                args.src_file = '/home/jhmoon/MedViLL/data/openi/Test.jsonl'
                print("OpenI data load")
            elif 'mimic' in re.split(r'[ _/]', args.model_recover_path):
                args.src_file = '/home/jhmoon/MedViLL/data/mimic/Test.jsonl'
                print("MimiC data load")
            elif 'pytorch_model' in re.split(r'[ _/.]', args.model_recover_path):
                args.src_file = '/home/jhmoon/MedViLL/data/mimic/Test.jsonl'
                print("MimiC data load")
            else: raise Exception("Path Error!!")
            
            img_dat = [json.loads(l) for l in open(args.src_file)]
            img_idx = 0
            radta_resample = [random.choice(img_dat) for itr in range(len(img_dat))]

            for src in radta_resample:
                change_path = src['img'].split('/')
                fixed_path = change_path[:-2]
                fixed_path = "/".join(fixed_path)
                static_path = change_path[-2:]
                static_path = "/".join(static_path)            
                if fixed_path == '/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch':
                    fixed_path = '/home/data_storage/mimic-cxr/dataset/image_preprocessing/re_512_3ch/'
                    img_path = fixed_path + static_path
                else: img_path=src['img']
                src_tk = os.path.join(img_path)
                imgid = str(src['id'])
                label = src['label']
                text = src['text']
                eval_lst.append((img_idx, imgid, src_tk, label, text)) # img_idx: index 0~n, imgid: studyID, src_tk: img_path
                img_idx += 1

            input_lines = eval_lst
            next_i = 0
            output_lines = [""] * len(input_lines)
            total_batch = math.ceil(len(input_lines) / args.batch_size)
            criterion =  nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
            total_score = []
            model.eval()
            print('start the caption evaluation...')
            with tqdm(total=total_batch) as pbar:
                while next_i < len(input_lines):
                    _chunk = input_lines[next_i:next_i + args.batch_size] 

                    buf_id = [x[0] for x in _chunk] # img id
                    buf = [x[2] for x in _chunk]  #img path
                    gt_report = [x[-1] for x in _chunk]
                    next_i += args.batch_size

                    instances = []
                    for instance in [(buf[x], args.len_vis_input, gt_report[x]) for x in range(len(buf))]:
                        for proc in bi_uni_pipeline:
                            instances.append(proc(instance))
                            
                    with torch.no_grad():
                        batch = batch_list_to_batch_tensors(
                            instances)
                        batch = [t.to(device) for t in batch]

                        input_ids, token_type_ids, position_ids, input_mask, task_idx, img, vis_pe, gt_token = batch                           
                        traces = model(img, vis_pe, input_ids,  token_type_ids, 
                                position_ids, input_mask, gt_token, device, task_idx=task_idx)
                        
                        
                        if args.beam_size > 1:
                            traces = {k: v.tolist() for k, v in traces.items()}
                            output_ids = traces['pred_seq']
                            
                        else:
                            output_ids = traces[0].tolist()
                            pred_score = traces[1][0].tolist()
                            prediction = traces[-1]
                            masked_lm_loss = criterion(prediction.transpose(1, 2).float(), gt_token)

                        for i in range(len(buf)):
                            w_ids = output_ids[i]
                            output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                            output_tokens = []
                            for t in output_buf:
                                if t in ("[SEP]", "[PAD]"):
                                    break
                                output_tokens.append(t)

                            output_sequence = ' '.join(detokenize(output_tokens))
                            output_lines[buf_id[i]] = output_sequence
                            if args.beam_size == 1:
                                ppl = torch.exp(masked_lm_loss)
                                total_score.append(ppl.item())
                            else:
                                pass

            if args.beam_size == 1:
                predictions = [{'image_id': tup[1], 'gt_caption': tup[-1], 'gt_label': tup[-2], 'gen_caption': output_lines[img_idx]} for img_idx, tup in enumerate(input_lines)]
                print("avg ppl: ",np.mean(total_score))
                a,b,c,d, metric_pos1 = language_eval_bleu(args.model_recover_path, str(round(np.mean(total_score), 2))+'ppl_'+str(bootstrap)+'th_test', predictions)
                max_a.append(a)
                max_b.append(b)
                max_c.append(c)
                max_d.append(d)

                results_dict['bleu1'].append(a)
                results_dict['bleu2'].append(b)
                results_dict['bleu3'].append(c)
                results_dict['bleu4'].append(d)
                results_dict['ppl'].append(np.mean(total_score))
                results_dict['accuracy'].append(round(metric_pos1[0], 3))
                results_dict['precision'].append(round(metric_pos1[1], 3))
                results_dict['recall'].append(round(metric_pos1[2], 3))
                results_dict['f1'].append(round(metric_pos1[3], 3))
            else:
                predictions = [{'image_id': tup[1], 'gt_caption': tup[-1], 'gt_label': tup[-2], 'gen_caption': output_lines[img_idx]} for img_idx, tup in enumerate(input_lines)]
                a,b,c,d, metric_pos1 = language_eval_bleu(args.model_recover_path, str(args.beam_size)+str('beam')+str(bootstrap)+'th_test', predictions)
                max_a.append(a)
                max_b.append(b)
                max_c.append(c)
                max_d.append(d)
                print("(micro) accuracy, precision, recall, f1,  for pos1: {}, {}, {}, {}".format(round(metric_pos1[0], 3), round(metric_pos1[1], 3), round(metric_pos1[2], 3), round(metric_pos1[3], 3)))#, round(metric_pos1[4], 3)))

                results_dict['bleu1'].append(a)
                results_dict['bleu2'].append(b)
                results_dict['bleu3'].append(c)
                results_dict['bleu4'].append(d)
                results_dict['ppl'].append(np.mean(total_score))
                results_dict['accuracy'].append(round(metric_pos1[0], 3))
                results_dict['precision'].append(round(metric_pos1[1], 3))
                results_dict['recall'].append(round(metric_pos1[2], 3))
                results_dict['f1'].append(round(metric_pos1[3], 3))
            save_file = f"all_random{args.random_bootstrap_testnum}_beam{args.beam_size}_{args.model_recover_path.split('/')[-2]}_{args.model_recover_path.split('.')[-2]}ep_gen.pickle"
            with open(save_file, 'wb') as f:
                pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("results_dict", results_dict)
            torch.cuda.empty_cache()

        # with open('random100_report_gen.pickle', 'rb') as f:
        #     b = pickle.load(f)

        # print(results_dict == b)
        # exit()

if __name__ == "__main__":
    # with open('/home/jhmoon/MedViLL/all_random30_beam1_base_bi_openi_2_10ep_gen.pickle', 'rb') as f:
    #     b = pickle.load(f)
    # print(b)
    # exit()

    main()
