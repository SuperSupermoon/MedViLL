"""BERT for report generation finetuning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import os
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")
print("START", now)
import sys
import logging
import glob
import math
import json
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertConfig, AutoConfig
from pytorch_pretrained_bert.model import BertForPreTrainingLossMask, BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from loader_utils import batch_list_to_batch_tensors
import data_loader
from data_parallel import DataParallelImbalance
import wandb
import utils
import pickle
from collections import defaultdict
import time

def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dataset", default='mimic-cxr', type=str, help=["mimic-cxr, openi"])
    parser.add_argument("--vqa_rad", default="all", type=str, choices=["all", "chest", "head", "abd"])
    parser.add_argument("--data_set", default="train", type=str, help="train | valid")
    parser.add_argument('--img_hidden_sz', type=int, default=2048, help="Whether to use amp for fp16")
    parser.add_argument('--random_bootstrap_testnum', type=int, default=30, help="Random bootstrap num of Iteration")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")

    parser.add_argument("--mlm_task", type=str, default=True, help="The model will train only mlm task!! | True | False")
    parser.add_argument("--vqa_eval", action='store_true',  help="vqa_eval | True | False")
    
    parser.add_argument("--train_batch_size",default=1,type=int,help="Total batch size for training.")
    parser.add_argument("--num_train_epochs",default=100,type=int,help="Total number of training epochs to perform.")
    parser.add_argument('--from_scratch', action='store_true', default = False,
                help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument("--img_encoding", type=str, default='fully_use_cnn', choices=['random_sample', 'fully_use_cnn'])
    parser.add_argument('--len_vis_input', type=int, default=256,
                    help="The length of visual token input") #visual token의 fixed length를 100이라 하면, <Unknown> token 100개가 되고, 100개의 word 생성 가능.
    parser.add_argument('--max_len_b', type=int, default=253, help="Truncate_config: maximum length of segment B.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")

    parser.add_argument('--max_pred', type=int, default=3,
                        help="Max tokens of prediction.")
    parser.add_argument('--s2s_prob', default=1, type=float,
                            help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
    parser.add_argument('--bi_prob', default=0, type=float,
                            help="Percentage of examples that are bidirectional LM.")
    parser.add_argument('--hidden_size', type=int, default=768)
    # parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--bar', default=False, type=str,help="True or False")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model. ex)'./pretrained_model/pytorch_model.bin'") # model load
    parser.add_argument("--output_dir",
                        default='/home/edlab/jhmoon/mimic_mv_real/mimic-cxr/downstream_model/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                       help="The output directory where the log will be written.")
    
    parser.add_argument('--img_postion', default = True,
                        help="It will produce img_position.")
    parser.add_argument("--do_train",
                        action='store_true', default = True,
                        help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.") # 3e-5
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', default = False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',default = False,
                        help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                            "0 (default value): dynamic loss scaling.\n"
                            "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true', default = False,
                        help="Whether to use amp for fp16")
    parser.add_argument('--wandb', action='store_true', default = False,
                        help="Whether to use wandb logging")
                        
    parser.add_argument('--new_segment_ids', default = False, action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Number of workers for the data loader.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")

    parser.add_argument('--image_root', type=str, default='../../data/mimic/re_512_3ch/Train')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'valid'])

    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--sche_mode', default='warmup_linear', type=str, help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--use_num_imgs', default=-1, type=int)
    parser.add_argument('--max_drop_worst_ratio', default=0, type=float)
    parser.add_argument('--drop_after', default=6, type=int)    
    parser.add_argument('--tasks', default='report_generation',help='report_generation | vqa')
    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")

    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.global_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])

    if args.model_recover_path !=None:
        args.exp_name = args.model_recover_path.split('/')[-2]
    else:args.exp_name = 'finetune_only'

    print('global_rank: {}, local rank: {}'.format(args.global_rank, args.local_rank))
        
    args.max_seq_length = args.max_len_b + args.len_vis_input + 3 # +3 for 2x[SEP] and [CLS]
    if args.model_recover_path !=None:
        args.config_path = args.model_recover_path.split('/')[:-1]
        args.config_path = ('/').join(args.config_path)+'/config.json'    
    else:
        args.config_path = '/home/edlab/jhmoon/mimic_mv_real/mimic-cxr/pre-train/base_PAR_36,128/pytorch_model.bin'.split('/')[:-1]
        args.config_path = ('/').join(args.config_path)+'/config.json'    

    if args.tasks=='vqa':
        args.src_file = '/home/data_storage/mimic-cxr/dataset/data_RAD'
        args.img_path = '/home/data_storage/mimic-cxr/dataset/vqa_image/vqa_512_3ch'
        args.train_dataset = '/home/data_storage/mimic-cxr/dataset/data_RAD/trainet.json'
        args.file_valid_jpgs = 'data/vqa_rad/vqa_rad_original_set.json'  
    else:
        if args.generation_dataset == 'mimic-cxr':
            args.src_file = 'data/mimic/Train.jsonl'
            args.file_valid_jpgs = 'data/mimic/Valid.jsonl'
        else:
            args.src_file = 'data/openi/Train.jsonl'
            args.file_valid_jpgs = 'data/openi/Valid.jsonl'

    print(" # PID :", os.getpid())
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_file),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    torch.distributed.init_process_group(backend='nccl', init_method = args.dist_url, world_size=args.world_size, rank=args.global_rank)
            
    logger.info("device: {} distributed training: {}, 16-bits training: {}".format(device,  bool(args.local_rank != -1), args.fp16))
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)
    torch.backends.cudnn.benchmark

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # fix random seed
    utils.set_seed(123)

    if args.wandb and utils.is_main_process():
        wandb.init(config=args, project='report_gen', entity='mimic-cxr',  name = args.exp_name, reinit=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    print("args.mask_prob",args.mask_prob)
    print("args.train_batch_size",args.train_batch_size)
    bi_uni_pipeline = [data_loader.Preprocess4Seq2seq(args, args.max_pred, args.mask_prob,
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, args.bar,
        new_segment_ids=args.new_segment_ids, truncate_config={
        'max_len_b': args.max_len_b, 'trunc_seg': args.trunc_seg, 'always_truncate_tail':
        args.always_truncate_tail}, mode="s2s", len_vis_input=args.len_vis_input, local_rank=args.local_rank, load_vqa_set=(args.tasks=='vqa'))]

    bi_uni_pipeline.append(data_loader.Preprocess4Seq2seq(args, args.max_pred, args.mask_prob,
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, args.bar,
        new_segment_ids=args.new_segment_ids, truncate_config={
        'max_len_b': args.max_len_b, 'trunc_seg': args.trunc_seg, 'always_truncate_tail':
        args.always_truncate_tail}, mode="bi", len_vis_input=args.len_vis_input,
        local_rank=args.local_rank, load_vqa_set=(args.tasks=='vqa')))

    train_dataset = data_loader.Img2txtDataset(args, args.data_set,
        args.src_file, args.image_root,args.split, args.train_batch_size,
        tokenizer, args.max_seq_length, file_valid_jpgs=args.file_valid_jpgs,
        bi_uni_pipeline=bi_uni_pipeline, use_num_imgs=args.use_num_imgs,
        s2s_prob=args.s2s_prob, # this must be set to 1.
        bi_prob=args.bi_prob, tasks=args.tasks)

    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, sampler=train_sampler, num_workers=args.num_workers,
        collate_fn=batch_list_to_batch_tensors, pin_memory=True)

    t_total = int(len(train_dataloader) * args.num_train_epochs * 1. /
                args.gradient_accumulation_steps)

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3 if args.tasks == 'report_generation' else 0

    mask_word_id, eos_word_ids, pad_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[PAD]"]) # index in BERT vocab: 103, 102, 0

    # BERT model will be loaded! from scratch
    if (args.model_recover_path is None):
        _state_dict = {} if args.from_scratch else None
        _state_dict = {}
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=_state_dict, args=args, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob, len_vis_input=args.len_vis_input, tasks=args.tasks)
            
        print("scratch model's statedict : ")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        global_step = 0
        print("The model will train from scratch")
        
    else:
        print("Task :",args.tasks, args.s2s_prob)
        for model_recover_path in glob.glob(args.model_recover_path.strip()):
            print("Recover path: ", model_recover_path)
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)        
            model_recover = torch.load(model_recover_path)

            for key in list(model_recover.keys()):
                model_recover[key.replace('enc.', ''). replace('mlm.', 'cls.')] = model_recover.pop(key)            
            
            global_step = 0
        
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, 
            state_dict=model_recover,
            args=args,num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, 
            relax_projection=relax_projection,
            config_path=args.config_path, 
            task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, 
            label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, 
            #cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob, 
            len_vis_input=args.len_vis_input, tasks=args.tasks)

        model.load_state_dict(model_recover, strict=False)

        print("The pretrained model loaded and fine-tuning.")
        del model_recover
        torch.cuda.empty_cache()

    model.to(device)
    if args.wandb and utils.is_main_process():
        wandb.watch(model)

    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    model = DDP(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters=True)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                        schedule=args.sche_mode,
                        t_total=t_total)
    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)))
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    results_dict = defaultdict(list)
    if args.vqa_eval:
        total_acc, closed_acc, open_acc = vqa_eval(args, device, logger, bi_uni_pipeline, tokenizer, model, results_dict)
    else:
        if args.do_train:
            logger.info("***** Running training *****")
            model.train()
            print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

            if recover_step:
                start_epoch = recover_step+1
                print("Recoverd epoch", start_epoch)
            else:start_epoch = 1

            for i_epoch in trange(start_epoch, args.num_train_epochs+1, desc="Epoch"):
                if args.local_rank >= 0:
                    train_sampler.set_epoch(i_epoch-1)
                iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
                nbatches = len(train_dataloader)
                total_vqa_acc, total_closed_acc, total_open_acc, train_loss = 0, 0, 0, []

                avg_loss, batch_count, num_close, num_open = 0, 0, 0, 0
                for step, batch in enumerate(iter_bar):
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, task_idx, img, vis_pe, ans_labels, ans_type, organ = batch
                    if args.fp16:
                        img = img.half()
                        vis_pe = vis_pe.half()

                    loss_tuple = model(img, vis_pe, input_ids, segment_ids,
                        input_mask, lm_label_ids, ans_labels, masked_pos=masked_pos,
                        masked_weights=masked_weights, task_idx=task_idx,
                        drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0,
                        ans_type=ans_type)

                    masked_lm_loss, vqa_loss, vqa_acc, closed_acc, open_acc = loss_tuple

                    batch_count += 1
                    if args.tasks == 'report_generation':
                        masked_lm_loss = masked_lm_loss.mean()
                        loss = masked_lm_loss
                    else:
                        vqa_loss = vqa_loss.mean()
                        loss = vqa_loss
                        total_vqa_acc += vqa_acc
                        if str(type(closed_acc)) == "<class 'float'>":
                            total_closed_acc += closed_acc
                            num_close += 1

                        if str(type(open_acc)) == "<class 'float'>":
                            total_open_acc += open_acc 
                            num_open += 1

                    iter_bar.set_description('Iter (loss=%5.3f)' %(loss.item()))
                    train_loss.append(loss.item())

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        lr_this_step = args.learning_rate * \
                            warmup_linear(global_step/t_total,
                                        args.warmup_proportion)
                        if args.fp16:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                if args.tasks == 'vqa':
                    # total_acc, closed_acc, open_acc = vqa_eval(args, test_dataloader, model)
                    print({"epoch total_acc": total_vqa_acc.item()/len(train_loss),
                        "epoch closed_acc": total_closed_acc/num_close,
                        "epoch open_acc": total_open_acc/num_open})
                logger.info(
                    "** ** * Saving fine-tuned model and optimizer ** ** * ")
                model_to_save = model.module if hasattr(
                    model, 'module') else model  # Only save the model it-self
                output_config_file = os.path.join(args.output_dir, 'config.json')
                
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                
                output_model_file = os.path.join(
                    args.output_dir, "model.{0}.bin".format(i_epoch))
                output_optim_file = os.path.join(
                    args.output_dir, "optim.{0}.bin".format(i_epoch))
                if args.global_rank in (-1, 0): # save model if the first device or no dist
                    torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                if args.world_size > 1:
                    torch.distributed.barrier()
            if args.tasks == 'vqa':
                total_acc, closed_acc, open_acc = vqa_eval(args, device, logger, bi_uni_pipeline, tokenizer, model, results_dict)

def vqa_eval(args, device, logger, bi_uni_pipeline, tokenizer, model, results_dict):
    test_dataset = data_loader.Img2txtDataset(args, 'test',
            args.src_file, args.image_root, args.split, 1,
            tokenizer, args.max_seq_length, file_valid_jpgs=args.file_valid_jpgs,
            bi_uni_pipeline=bi_uni_pipeline, use_num_imgs=args.use_num_imgs,
            s2s_prob=args.s2s_prob, # this must be set to 1.
            bi_prob=args.bi_prob, tasks=args.tasks)
    test_sampler = RandomSampler(test_dataset, replacement=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler,num_workers=args.num_workers, shuffle=False, collate_fn=batch_list_to_batch_tensors,drop_last=True)

    for itr in range(args.random_bootstrap_testnum):
        logger.info("***** Running test *****")
        model.eval()
        print("Random bootstrap iter: ", itr+1)
        print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
        # iter_bar = tqdm(test_dataloader, desc='Eval Iter (loss=X.XXX)')
        total_vqa_acc, total_closed_acc, total_open_acc, test_loss = 0, 0, 0, []
        avg_loss, batch_count, num_close, num_open = 0, 0, 0, 0
        start = time.time()

        with torch.no_grad(): 
            for step, batch in enumerate(test_dataloader):
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, task_idx, img, vis_pe, ans_labels, ans_type, organ = batch
                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()

                loss_tuple = model(img, vis_pe, input_ids, segment_ids,
                    input_mask, lm_label_ids, ans_labels, masked_pos=masked_pos,
                    masked_weights=masked_weights, task_idx=task_idx,
                    drop_worst_ratio=args.max_drop_worst_ratio,
                    ans_type=ans_type)

                _, vqa_loss, vqa_acc, closed_acc, open_acc = loss_tuple

                batch_count += 1
                vqa_loss = vqa_loss.mean()
                loss = vqa_loss
                total_vqa_acc += vqa_acc
                if str(type(closed_acc)) == "<class 'float'>":
                    total_closed_acc += closed_acc
                    num_close += 1

                if str(type(open_acc)) == "<class 'float'>":
                    total_open_acc += open_acc 
                    num_open += 1

                test_loss.append(loss.item())
        
        results_dict['total_acc'].append((total_vqa_acc.item()/len(test_loss)))
        results_dict['closed_acc'].append((total_closed_acc/num_close))
        results_dict['open_acc'].append((total_open_acc/num_open))
        logger.info("***** CUDA.empty_cache() *****")
        if (args.model_recover_path is None):
            save_file = f"vqa_tr_random{args.random_bootstrap_testnum}_finetune_only_50ep.pickle"
        elif args.model_recover_path.split('/')[-1].split('.')[0] == 'pytorch_model':
            save_file = f"final_vqa_tr_ft_random{args.random_bootstrap_testnum}_{args.model_recover_path.split('/')[-3]}_{args.model_recover_path.split('/')[-2]}_{args.vqa_rad}_100ep.pickle"
        else:
            save_file = f"vqa_random{args.random_bootstrap_testnum}_{args.model_recover_path.split('/')[-2]}_{args.model_recover_path.split('.')[-2]}ep.pickle"
        with open(save_file, 'wb') as f:
            pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("results_dict", results_dict)
        torch.cuda.empty_cache()
        end = time.time()
        print(end - start)
        
    return (total_vqa_acc/len(test_loss))*100, (total_closed_acc/num_close)*100, (total_open_acc/num_open)*100

if __name__ == "__main__":
    main()
