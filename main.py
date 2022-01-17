"""
MedViLL, pre-training model main run.py
"""
import os
import argparse
from datetime import datetime

from data.dataset_origin import CXRDataset
from torch.utils.data import DataLoader

from utils.utils import *
from models.train_origin import CXRBERT_Trainer  # CXR-BERT

from transformers import BertTokenizer, AlbertTokenizer, AutoTokenizer

def train(args):
    set_seed(args.seed)

    # TODO: bert-base,small,tiny tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True).tokenize

    print("Load Train dataset", args.train_dataset)
    train_dataset = CXRDataset(args.train_dataset, tokenizer, args)

    print("Load Test dataset", args.test_dataset)
    test_dataset = CXRDataset(args.test_dataset, tokenizer, args) \
        if args.test_dataset is not None else None

    print("Create DataLoader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
        if test_dataset is not None else None

    print("Creating BERT Trainer")
    trainer = CXRBERT_Trainer(args, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    print("Training Start!")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=str,
                        default='data/mimic/Train.jsonl',
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str,
                        default='data/mimic/Valid.jsonl',
                        help='test dataset for evaluating train set')

    output_path = 'output/' + str(datetime.now())
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)

    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n inter: setting n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--mlm_task", type=str, default=True,
                        help="The model will train only mlm task!! | True | False")
    parser.add_argument("--itm_task", type=str, default=True,
                        help="The model will train only itm task!! | True | False")

    parser.add_argument('--attn_1d', type=bool, default=False, help='choose 1d attn(True) or full attn(False)')
    parser.add_argument('--BAR_attn', default=True, type=bool, help="Bidirectional Auto Regressive attn mask")
    parser.add_argument('--Mixed', default=False, type=bool, help="Mixed attn mask")
    parser.add_argument('--s2s_prob', default=1.0, type=float, help="S2S attention prob.")
    parser.add_argument('--bi_prob', default=0.0, type=float,  help="Full_attention prob.")
    parser.add_argument('--disturbing_mask', default=False, type=bool, help="Baseline attn mask(I-I, T-T)")

    parser.add_argument("--epochs", type=int, default=50, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=36, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=20, help="dataloader worker size")

    # TODO: init model
    parser.add_argument("--hidden_size", type=int, default=768, choices=[768, 512, 128])
    parser.add_argument("--embedding_size", type=int, default=768, choices=[768, 512, 128])

    ## pre_trained_model_path, weight_load
    parser.add_argument("--weight_load", type=bool, default=False, help='pre-trained_model_mid_epoch_load')
    parser.add_argument("--pre_trained_model_path", type=str, default='/home/cxr-bert/clinicalbert_vlp_re35_5')
    parser.add_argument("--bert_model", type=str, default="bert-base-scratch")  # pre-trained CXR-BERT
    parser.add_argument("--vocab_size", type=int, default=30522)

    parser.add_argument("--img_postion", default=True, help='img_postion use!')
    parser.add_argument("--seq_len", type=int, default=253, help="maximum sequence len")
    parser.add_argument("--max_seq_len", type=int, default=512, help="total sequence len")
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_encoder", type=str, default='random-pixel', choices=['random-pixel', 'full-fiber', 'ViT'])
    parser.add_argument("--img_channel", type=int, default=3)
    parser.add_argument("--num_image_embeds", type=int, default=180, choices=[180, 256])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of AdamW")  # 0.01 , AdamW

    args = parser.parse_args()

    train(args)
