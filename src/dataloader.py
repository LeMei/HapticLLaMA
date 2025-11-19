# -*- encoding:utf-8 -*-
# from nis import cat
from os import access
import random
import numpy as np
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
from create_dataset import HAPTIC
from config import DEVICE, Config
import warnings
import json as js
from transformers import LlamaTokenizer,AutoTokenizer
from config import get_args

args = get_args()
mode = args.mode

tokenizer = AutoTokenizer.from_pretrained("../data/Haptic/pretrain/updated_llama_tokenizer_{}.pt".format(mode))

class HapticsDataset(Dataset):
    def __init__(self, args, config):

        dataset_name = args.dataset
        self.dataset_name = str.upper(dataset_name)
        dataset = globals()[self.dataset_name](config)

        self.data = dataset.get_data(config.mode)
        self.data_len = len(self.data)

    @property
    def ta_dim(self):
        return 768, 768

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data_len


def get_loader(args, config, truncate=False, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = HapticsDataset(args,config)
    print('mode:{}'.format(config.mode))

    if config.mode == 'train':
        args.n_train = len(dataset)
    elif config.mode == 'valid':
        args.n_valid = len(dataset)
    elif config.mode == 'test':
        args.n_test = len(dataset)

    def collate_fn(batch):

        signal_ids = []
        haptics = []
        labels = []
        categories = []
        prompts = []

        for sample in batch:
            signal_id, haptic, label, category = sample[0], sample[1], sample[2],sample[3]
            signal_ids.append(signal_id)
            labels.append(label.strip()+'<eos>')
            haptics.append(' '.join(haptic))
            categories.append(category.strip())
            prompts.append("the {} description is:".format(category.strip()))

        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(haptics, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids
        input_atts = inputs.attention_mask

        encoding = tokenizer(labels, padding=True, truncation=True, return_tensors="pt")
        labels_ids = encoding.input_ids
        labels_atts = encoding.attention_mask

        prompt_enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        prompt_ids = prompt_enc.input_ids
        prompt_atts = prompt_enc.attention_mask

        post_input_ids = torch.cat((input_ids, prompt_ids, labels_ids),dim=1)
        post_input_atts = torch.cat((input_atts,prompt_atts, labels_atts),dim=1)

        post_labels_ids = torch.cat((input_ids, prompt_ids,labels_ids),dim=1)
        post_labels_atts = torch.cat((input_atts, prompt_atts,labels_atts),dim=1)


        prompt_ids = torch.cat((input_ids,prompt_ids),dim=1)
        prompt_atts = torch.cat((input_atts,prompt_atts),dim=1)

        start_idx = prompt_ids.shape[1]

        post_labels_ids[:,:start_idx]=-100
        post_labels_ids[post_labels_ids == tokenizer.pad_token_id] = -100


        # batch_size = input_ids.shape[0]
        # bos_tokens = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long)
        # post_input_ids = torch.cat([bos_tokens,post_input_ids],dim=1)

        # post_input_ids[:,:start_idx].detach()
        # breakpoint()
        

        # input_ids = torch.LongTensor(haptics)
        # print('inputs_ids:',input_ids)
        return signal_ids, (post_input_ids,post_input_atts), (post_labels_ids,post_labels_atts), (input_ids,input_atts), (prompt_ids, prompt_atts), categories, labels, 

        # return signal_ids, (input_ids,input_atts), (labels_ids,labels_atts), categories
    
    data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn)
    
    return data_loader