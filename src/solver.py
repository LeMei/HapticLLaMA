global logger

import re
from tkinter import NO
import torch
from torch import nn, std_mean
import sys
import json
import torch.optim as optim
import numpy as np
import time
import random
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoTokenizer

from model import Model
from modules.lora_layer import LoRALinear
from utils.eval_metrics import *
from utils.util import read_json,calculate_average_metric
from config import get_args, get_config, DEVICE


class Solver(object):
    def __init__(self, args, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):

        self.args = args
        
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        # initialize the model
        if model is None:
            self.model = self.init_model(args)

        self.batch_size = args.batch_size
        self.dataset = args.dataset
        
        # initialize the device
        self.device = DEVICE
        self.n_gpu = 1
        self.model.to(self.device)

        ground_data_path = args.ground_data
        # breakpoint()
        self.ground_data = read_json(ground_data_path)


        self.optimizer_main, self.scheduler_main = self.prep_optimizer(args,self.device, self.n_gpu)
        # print('pre_optimizer',self.pre_optimizer)
        # print('optimizer',self.optimizer)


        # self.optimizer, self.scheduler = self.prep_optimizer(args, num_train_optimization_steps,self.device, self.n_gpu)

    def set_seed_logger(self, args): 
    # predefining random initial seeds
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  
        torch.cuda.set_device(args.local_rank) 
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        # logger = get_logger(os.path.join(args.output_dir, "log.txt"))
        if args.local_rank == 0:
            logger.info("Effective parameters:")
            for key in sorted(args.__dict__):
                logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

        return args

    def init_model(self, args):
        model = Model(args=args)
        return model

    def prep_optimizer(self, args, device, n_gpu, local_rank=None, coef_lr=1.):

        main_param = []
        for name, p in self.model.named_parameters():
            p.requires_grad = False
        
        for name, param in self.model.named_parameters():
            # if ("lora_" in name and 'layers.15' in name) :
            print(name)
            if ("lora_" in name):
                param.requires_grad = True
            if "embed" in name:
                param.requires_grad = True

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(name)
                main_param.append(param)

        no_decay = ['bias', 'LayerNorm.weight','LayerNorm.bias']
        optimizer_grouped_parameters = [
            {'params': main_param, 'weight_decay': self.args.weight_decay_main, 'lr': self.args.learn_rate}
        ]

        optimizer = getattr(torch.optim, self.args.optimizer)(
            optimizer_grouped_parameters
        )
            
        scheduler_main = ReduceLROnPlateau(optimizer, mode='min', patience=self.args.when, factor=0.5, verbose=True)

        return optimizer, scheduler_main

    # def save_model(self, args, model, epoch):
    #     # Only save the model it-self
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     output_model_file = os.path.join(
    #         args.output_dir, "pytorch_model_{}.bin.".format(epoch))
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     logger.info("Model saved to %s", output_model_file)
    #     return output_model_file
    
    def save_model(self, epoch, args, device, version, model_file_url=None):
        if model_file_url is None or len(model_file_url) == 0:
            model_file_url = os.path.join(args.output_dir, "{}_full_lora_{}_{}_{}.pth".format(args.mode, epoch, version, args.suffix))

        # lora_state_dict = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
        torch.save(self.model.state_dict(), model_file_url)
        return model_file_url


    def train_and_eval(self):

        model = self.model
        scheduler = self.scheduler_main

        def train(model, optimizer=None):
            epoch_loss = 0.0
            model.train()
            num_batches = self.args.n_train // self.batch_size

            # print('num_batches:{}'.format(num_batches))

            for step, batch_data in enumerate(self.train_loader):
                signal_id, (input_ids,input_atts),(label_ids,label_atts),(signal_input_ids,signal_input_atts), (prompt_ids, prompt_atts), category, ground_caption = batch_data

                model.zero_grad()
                with torch.cuda.device(0):
                    input_ids,input_atts,label_ids,label_atts = input_ids.to(self.device),input_atts.to(self.device), label_ids.to(self.device),label_atts.to(self.device)
                
                    batch_loss, logits, hidden_states = self.model(inputs = input_ids,att_mask = input_atts, labels=label_ids)
                    # breakpoint()
                    epoch_loss = epoch_loss + batch_loss
                    # batch_loss = batch_loss.requires_grad_(True)
                    batch_loss.backward()
                    self.optimizer_main.step()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

                    self.optimizer_main.step()
                    self.optimizer_main.zero_grad()

            return epoch_loss / self.args.n_train, epoch_loss
        
        def evaluate(model, loader, test=False):
            # loader = self.test_loader if test else self.dev_loader
            model.eval()
            epoch_loss = 0.0
            pred_tokens = []
            truth_tokens = []
            bleu1_list, bleu4_list = [], []
            rouge_meteor_list = []
            category_list = []
            
            with torch.no_grad():
                for i_batch, batch_data in enumerate(loader):
                    signal_id, (input_ids,input_atts),(label_ids,label_atts),(signal_input_ids,signal_input_atts),(prompt_ids, prompt_atts),category,ground_caption = batch_data

                    batch_size = input_ids.shape[0]
                    
                    bleu1_batch, bleu4_batch = [], []
                    rouge_meteor_batch = []

                    with torch.cuda.device(0):
                        input_ids,input_atts,label_ids,label_atts = input_ids.to(self.device),input_atts.to(self.device), label_ids.to(self.device),label_atts.to(self.device)

                        signal_input_ids, signal_input_atts = signal_input_ids.to(self.device), signal_input_atts.to(self.device)
                        prompt_ids,prompt_atts = prompt_ids.to(self.device),prompt_atts.to(self.device)

     
                        batch_loss, logits, hidden_states = self.model(inputs = input_ids,att_mask = input_atts, labels=label_ids)
                        epoch_loss = epoch_loss + batch_loss
                        if test:
                            # breakpoint()
                            # #v1#
                            # generated_tokens = self.model.generate(inputs = signal_input_ids,input_atts=signal_input_atts)
                            #v2#
                            generated_tokens = self.model.generate(inputs = prompt_ids,input_atts=prompt_atts)
                            # print('generated_caption:{}\n'.format(generated_tokens))
                            for i in range(batch_size):
                                print('---------------------{}:{}-------------------'.format(signal_id[i],category[i])) 
                                # generated_caption = generated_tokens[i].split('.')[0]+'.'
                                # print('generated_caption:{}\n'.format(generated_caption))
                                # if '<eos>' in generated_tokens[i]:
                                #     generated_caption = generated_tokens[i].split('<eos>')[0]
                                
                                token = generated_tokens[i]
                                match = re.search(r"<eos>?|<eo", token)
                                if match:
                                    generated_caption = token[:match.start()]
                                elif '.' in generated_tokens[i]:
                                    generated_caption = generated_tokens[i].split('.')[0]+'.'
                                else:
                                    generated_caption = generated_tokens[i].split('and')[0]+'.'

                                print('generated_caption:{}\n'.format(generated_tokens[i]))
                                print('ground_caption:{}\n'.format(ground_caption[i]))
                                ground_dict = self.ground_data[signal_id[i]]
                                ground_des = ground_dict[category[i]]

                                # max_bleu_1, min_bleu_1, max_bleu_4, min_bleu_4 = compute_bleu(ground_des,generated_caption)
                                avg_bleu_1, std_bleu_1, avg_bleu_4, std_bleu_4 = compute_avg_bleu(ground_des,generated_caption)
                                rouge,rouge_std = compute_rouge(ground_des,generated_tokens[i])
                                meteor = compute_meteor(ground_des, generated_tokens[i])

                                print('signal id:{},avg_bleu_1:{}, std_bleu_1:{}, avg_bleu_4:{}, std_bleu_4:{}, rouge:{}, meteor:{}\
                                      '.format(signal_id[i], avg_bleu_1, std_bleu_1, avg_bleu_4, std_bleu_4, rouge, meteor))
                                
                                bleu1_batch.append([avg_bleu_1, std_bleu_1])
                                bleu4_batch.append([avg_bleu_4, std_bleu_4])

                                rouge_meteor_batch.append([rouge,meteor])
                                
                                pred_tokens.append(generated_tokens[i])
                                truth_tokens.extend(ground_des)
                            
                            bleu1_list.extend(bleu1_batch)
                            bleu4_list.extend(bleu4_batch)

                            rouge_meteor_list.extend(rouge_meteor_batch)
                            category_list.extend(category)
                            
            if not test:
                return epoch_loss / self.args.n_valid, epoch_loss
            else:
                return epoch_loss / self.args.n_valid, epoch_loss, pred_tokens, truth_tokens, bleu1_list, bleu4_list, rouge_meteor_list, category_list
                        

        best_result = {
            'max_bleu1':-1,
            'min_bleu1':-1,
            'min_bleu4':-1,
            'max_bleu4':-1,
            'rouge':-1,
            'meteor':-1,
            'bleu1':{},
            'bleu2':{},
            'category_rouge':{},
            'category_meteor':{}
        }

        for epoch in range(1, self.args.num_epochs+1):
            start = time.time()
            avg_train_loss, train_loss = train(model)
            print('start:{}, epoch:{}, avg train loss:{}, train loss:{}\n'.format(start, epoch, avg_train_loss, train_loss))

            # if self.dev_loader:
            #     avg_valid_loss, valid_loss = evaluate(model, self.dev_loader, test=False)
            # else:
            #     avg_valid_loss, valid_loss = 0, 0


            # breakpoint()
            avg_test_loss, test_loss, generated_tokens, truth_tokens, bleu1_list, bleu4_list, rouge_meteor_list, category_list = evaluate(model, self.test_loader,test=True)

            avg_bleu1, avg_bleu4, avg_rouge_score, avg_meteor_score, category_avg_bleu1,category_avg_bleu4,category_avg_rouge_meteor\
            = calculate_average_metric(bleu1_list=bleu1_list, bleu4_list=bleu4_list,rouge_meteor_list=rouge_meteor_list, category_list=category_list)
            avg_max_bleu1, avg_min_bleu1 = avg_bleu1[0],avg_bleu1[1]
            avg_max_bleu4, avg_min_bleu4 = avg_bleu4[0],avg_bleu4[1]

            if avg_max_bleu4 > best_result['max_bleu4']:
                best_result['max_bleu4'] = avg_max_bleu4
                best_result['max_bleu1'] = avg_max_bleu1
                best_result['min_bleu4'] = avg_min_bleu4
                best_result['min_bleu1'] = avg_min_bleu1
                best_result['rouge'] = avg_rouge_score
                best_result['meteor'] = avg_meteor_score
                best_result['epoch'] = epoch
                best_result['bleu1'] = category_avg_bleu1
                best_result['bleu4'] = category_avg_bleu4
                best_result['category_rouge_meteor'] = category_avg_rouge_meteor
                self.save_model(epoch, self.args, version='5.1', device=DEVICE)

                
            print('epoch:{}, avg_max_bleu1:{:.4f}, avg_min_bleu1:{:.4f}, avg_max_bleu4:{:.4f}, avg_min_bleu4:{:.4f}, avg_rouge:{:.4f}, avg_meteor:{:.4f}'.format(epoch, avg_max_bleu1, avg_min_bleu1,\
                                                                                                                               avg_max_bleu4, avg_min_bleu4, avg_rouge_score, avg_meteor_score))

        print('best_epoch:{}, best_max_bleu1:{:.4f}, best_min_bleu1:{:.4f}, best_max_bleu4:{:.4f}, best_min_bleu4:{:.4f}, best_rouge:{:.4f}, best_meteor:{:.4f}\
              '.format(best_result['epoch'], best_result['max_bleu1'], best_result['min_bleu1'],best_result['max_bleu4'], best_result['min_bleu4'], best_result['rouge'], best_result['meteor']))
        
        print('start outputing best_category_bleu1:')

        for k in best_result['bleu1'][0].keys():
            print('{}--bleu1_min:{:.4f},bleu1_max:{:.4f}'.format(k,best_result['bleu1'][0][k],best_result['bleu1'][1][k]))

        print('start outputing best_category_bleu4:')

        for k in best_result['bleu4'][0].keys():
            print('{}--bleu4_min:{:.4f},bleu4_max:{:.4f}'.format(k,best_result['bleu4'][0][k],best_result['bleu4'][1][k]))

            
        print('start outputing best_category_rouge_meteor:')

        for k in best_result['category_rouge_meteor'][0].keys():
            print('{}--rouge:{:.4f},meteor:{:.4f}'.format(k,best_result['category_rouge_meteor'][0][k],best_result['category_rouge_meteor'][1][k]))
        

        # print(generated_tokens)
        # print('start:{}, epoch:{}, avg train loss:{}, train loss:{}, avg valid loss:{}, valid loss:{}, avg test loss:{}, test loss:{}'.format(start, epoch, avg_train_loss, train_loss, \
        # avg_valid_loss, valid_loss, avg_test_loss, test_loss))


        sys.stdout.flush()


            





            
        


