"""
MedViLL, pre-training model main run.py
"""
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

from data.dataset_origin import create_dataset, create_sampler, create_loader

from utils import utils
from utils.utils import *
from models.train_origin import MedViLL_Trainer  # CXR-BERT

from transformers import AutoTokenizer

def train(config, args):
    init_distributed_mode(args)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    print("Load Train dataset", config['train_dataset'])
    dset  = create_dataset(tokenizer=tokenizer, args=args, config=config)

    print("Create DataLoader")
    if args.distributed:
        samplers = create_sampler(dset, [True, False, False], utils.get_world_size(), utils.get_rank())         
    else:
        samplers = [None, None, None]
        
    train_data_loader, _, test_data_loader = create_loader(dset, samplers, batch_size=[config['batch_size'],config['batch_size'],config['batch_size']], is_trains=[True, False, False], num_workers=0)

    print("Creating BERT Trainer")
    trainer = MedViLL_Trainer(args, config, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    print("Training Start!")
    for epoch in range(config['epochs']):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm_task", type=str, default=True)
    parser.add_argument("--itm_task", type=str, default=True)

    parser.add_argument('--BAR_attn', default=True, type=bool, help="Bidirectional Auto Regressive attn mask")
    parser.add_argument('--Mixed', default=False, type=bool, help="Mixed attn mask")
    parser.add_argument('--s2s_prob', default=1.0, type=float, help="S2S attention prob.")
    parser.add_argument('--bi_prob', default=0.0, type=float,  help="Full_attention prob.")
    parser.add_argument('--disturbing_mask', default=False, type=bool, help="Baseline attn mask(I-I, T-T)")

    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    ## pre_trained_model_path, weight_load
    parser.add_argument("--weight_load", type=bool, default=False, help='pre-trained_model_mid_epoch_load')
    parser.add_argument("--pre_trained_model_path", type=str)
    
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    
    config = yaml.load(open('./configs/pretrain.yaml', 'r'), Loader=yaml.Loader)
    now = datetime.datetime.now()
    nowDate = now.strftime('%m%d-%H%M')
    args.output_dir = 'output/'+nowDate
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train(config, args)


