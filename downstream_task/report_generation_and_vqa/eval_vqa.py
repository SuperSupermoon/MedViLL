"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
import sys

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from misc.data_parallel import DataParallelImbalance
from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader

from vlp.lang_utils import language_eval

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]


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

    # General
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")

    # For decoding
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--max_tgt_length', type=int, default=20,
                        help="maximum length of target sequence")

    # Others for VLP
    parser.add_argument("--src_file", default='/mnt/dat/COCO/annotations/dataset_coco.json', type=str,		
                        help="The input data file name.")		
    parser.add_argument("--ref_file", default='pythia/data/v2_mscoco_val2014_annotations.json', type=str,
                        help="The annotation reference file name.")
    parser.add_argument('--dataset', default='coco', type=str,
                        help='coco | flickr30k | cc')
    parser.add_argument('--len_vis_input', type=int, default=100)
    # parser.add_argument('--resnet_model', type=str, default='imagenet_weights/resnet101.pth')
    parser.add_argument('--image_root', type=str, default='/mnt/dat/COCO/images')		
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5', type=str)
    parser.add_argument('--region_det_file_prefix', default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument("--output_dir",
                        default='tmp',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--file_valid_jpgs', default='', type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    args.max_seq_length = args.max_tgt_length + args.len_vis_input + 3 # +3 for 2x[SEP] and [CLS]
    tokenizer.max_len = args.max_seq_length

    bi_uni_pipeline = []
    bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(0, 0,
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        new_segment_ids=args.new_segment_ids, truncate_config={
        'max_len_b': args.max_tgt_length, 'trunc_seg': 'b', 'always_truncate_tail': True},
        mode="bi", len_vis_input=args.len_vis_input,
        region_bbox_file=args.region_bbox_file, region_det_file_prefix=args.region_det_file_prefix,
        load_vqa_ann=True)]


    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    logger.info('Attempting to recover models from: {}'.format(args.model_recover_path))
    if 0 == len(glob.glob(args.model_recover_path.strip())):
        logger.error('There are no models to recover. The program will exit.')
        sys.exit(1)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, task_idx=0,
            max_position_embeddings=512, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(-1),
            drop_prob=args.drop_prob, enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, tasks='vqa2')
        del model_recover

        model.to(device)

        torch.cuda.empty_cache()
        model.eval()

        eval_lst = []
        img_dat = np.load(args.src_file, allow_pickle=True)
        img_idx = 0
        for i in range(1, img_dat.shape[0]):
            if args.enable_butd:
                src_tk = os.path.join(args.image_root, img_dat[i]['image_name'].split('_')[1],
                    img_dat[i]['feature_path'])
            else:
                raise NotImplementedError
            tgt_tk = tokenizer.tokenize(img_dat[i]['question_str'])
            eval_lst.append((img_idx, src_tk, tgt_tk, img_dat[i]['question_id']))
            img_idx += 1
        input_lines = eval_lst

        next_i = 0
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)
        predictions = []

        print('start the VQA evaluation...')
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf = [(x[1], x[2]) for x in _chunk]
                buf_id = [(x[0], x[3]) for x in _chunk]
                next_i += args.batch_size
                instances = []
                for instance in buf:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance[:2]+({'answers': ['dummy']},)))
                with torch.no_grad():
                    batch = batch_list_to_batch_tensors(
                        instances)
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, img, vis_masked_pos, vis_pe, _ = batch

                    conv_feats, _ = cnn(img.data) # Bx2048x7x7
                    conv_feats = conv_feats.view(conv_feats.size(0), conv_feats.size(1),
                        -1).permute(0,2,1).contiguous()

                    ans_idx = model(conv_feats, vis_pe, input_ids, segment_ids,
                        input_mask, lm_label_ids, None, is_next, masked_pos=masked_pos,
                        masked_weights=masked_weights, task_idx=task_idx,
                        vis_masked_pos=vis_masked_pos, drop_worst_ratio=0,
                        vqa_inference=True)

                    for ind, (eval_idx, ques_id) in enumerate(buf_id):
                        predictions.append({'question_id': ques_id, 'answer': bi_uni_pipeline[0].ans_proc.idx2word(ans_idx[ind])})

                pbar.update(1)

        results_file = os.path.join(args.output_dir, 'vqa2-results-'+args.model_recover_path.split('/')[-2]+'-'+args.split+'-'+args.model_recover_path.split('/')[-1].split('.')[-2]+'.json')
        json.dump(predictions, open(results_file, 'w'))

if __name__ == "__main__":
    main()