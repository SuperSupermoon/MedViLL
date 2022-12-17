"""
Construct CXR-BERT or BertForMaskedLM, Training and Saving
"""
import os
import tqdm
import torch
import datetime
import torch.nn as nn
import numpy as np

from models.MedViLL_origin import MedViLL

from transformers.optimization import AdamW
from transformers import BertConfig, AlbertConfig, AutoConfig


class MedViLL_Trainer():
    def __init__(self, args, configs, train_dataloader, test_dataloader=None):
        self.args = args
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Current cuda device ', torch.cuda.current_device())  # check

        if args.weight_load:
            model_config = AutoConfig.from_pretrained(args.pre_trained_model_path)
            model_state_dict = torch.load(os.path.join(args.pre_trained_model_path, 'pytorch_model.bin'))
            self.model = MedViLL.from_pretrained(args.pre_trained_model_path, state_dict=model_state_dict,
                                model_config=model_config, args=args, configs=configs).to(self.device)
            print('resume')
            print(model_config)
        else:
            model_config = BertConfig.from_pretrained("bert-base-uncased")
            self.model = MedViLL(model_config, args, configs).to(self.device)

        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)

        self.model_without_ddp = self.model
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            self.model_without_ddp = model.module
        
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optimizer = AdamW(self.model.parameters(), lr=self.configs['lr'])
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.itm_criterion = nn.CrossEntropyLoss()
        self.step_cnt = 0
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        train_losses, train_itm_loss, train_mlm_loss = [], [], []
        train_data_iter = tqdm.tqdm(enumerate(self.train_data),
                                    desc=f'EP_:{epoch}',
                                    total=len(self.train_data),
                                    bar_format='{l_bar}{r_bar}')
        
        total_correct, total_element, total_mlm_correct, total_mlm_element = 0,0,0,0
        total_valid_correct, total_valid_element, total_mlm_valid_correct, total_mlm_valid_element = 0,0,0,0

        for i, data in train_data_iter:
            cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data
            
            cls_tok = cls_tok.to(self.device)
            input_ids = input_ids.to(self.device)
            txt_labels = txt_labels.to(self.device)
            attn_masks = attn_masks.to(self.device)
            img = img.to(self.device)
            segment = segment.to(self.device)
            is_aligned = is_aligned.to(self.device)
            sep_tok = sep_tok.to(self.device)
            
            mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)

            if self.args.mlm_task and self.args.itm_task == False:
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                loss = mlm_loss
                print('only mlm_loss')

            if self.args.itm_task and self.args.mlm_task == False:
                itm_loss = self.itm_criterion(itm_output, is_aligned)
                loss = itm_loss
                print('only itm_loss')

            if self.args.mlm_task and self.args.itm_task:
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                train_mlm_loss.append(mlm_loss.item())
                itm_loss = self.itm_criterion(itm_output, is_aligned)
                train_itm_loss.append(itm_loss.item())
                loss = itm_loss + mlm_loss

            train_losses.append(loss.item())
            self.optimizer.zero_grad()  # above
            loss.backward()
            self.optimizer.step()

            if self.args.itm_task:
                correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                total_correct += correct
                total_element += is_aligned.nelement()

            if self.args.mlm_task:
                eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                txt_labels_np = txt_labels.cpu().numpy()
                for bs, label in enumerate(txt_labels_np):
                    index = np.where(label == -100)[0]
                    f_label = np.delete(label, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_correct += f_eq.sum()
                    total_mlm_element += len(f_label)

        print("avg loss per epoch", np.mean(train_losses))
        print("avg itm acc per epoch", round(total_correct / total_element * 100, 3))
        test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                                   desc=f'EP_:{epoch}',
                                   total=len(self.test_data),
                                   bar_format='{l_bar}{r_bar}')
        self.model.eval()
        with torch.no_grad():
            eval_losses, eval_mlm_loss, eval_itm_loss = [], [], []            
            for i, data in test_data_iter:
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data
                cls_tok = cls_tok.to(self.device)
                input_ids = input_ids.to(self.device)
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                img = img.to(self.device)
                segment = segment.to(self.device)
                is_aligned = is_aligned.to(self.device)
                sep_tok = sep_tok.to(self.device)

                mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)

                if self.args.mlm_task and self.args.itm_task == False:
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_loss = valid_mlm_loss
                    print('only valid mlm loss')

                if self.args.itm_task and self.args.mlm_task == False:
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    valid_loss = valid_itm_loss
                    print('only valid itm loss')

                if self.args.mlm_task and self.args.itm_task:
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    eval_mlm_loss.append(valid_mlm_loss.item())
                    eval_itm_loss.append(valid_itm_loss.item())
                    valid_loss = valid_itm_loss + valid_mlm_loss

                eval_losses.append(valid_loss.item())

                if self.args.itm_task:
                    valid_correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                    total_valid_correct += valid_correct
                    total_valid_element += is_aligned.nelement()

                if self.args.mlm_task:
                    eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                    txt_labels_np = txt_labels.cpu().numpy()
                    for bs, label in enumerate(txt_labels_np):
                        index = np.where(label == -100)[0]
                        f_label = np.delete(label, index)
                        f_eq = np.delete(eq[bs], index)
                        total_mlm_valid_correct += f_eq.sum()
                        total_mlm_valid_element += len(f_label)

            print("avg loss in testset", np.mean(eval_losses))
            print("avg itm acc in testset", round(total_valid_correct / total_valid_element * 100, 3))

    def save(self, epoch, file_path):
        save_path_per_ep = os.path.join(file_path, str(epoch))
        if not os.path.exists(save_path_per_ep):
            os.mkdir(save_path_per_ep)
            os.chmod(save_path_per_ep, 0o777)

        if torch.cuda.device_count() > 1:
            self.model.module.save_pretrained(save_path_per_ep)
            print(f'Multi_EP: {epoch} Model saved on {save_path_per_ep}')
        else:
            self.model.save_pretrained(save_path_per_ep)
            print(f'Single_EP: {epoch} Model saved on {save_path_per_ep}')
        os.chmod(save_path_per_ep + '/pytorch_model.bin', 0o777)
