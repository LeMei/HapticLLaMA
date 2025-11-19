import json
from transformers import AutoTokenizer
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from utils.util import load_json

from config import get_args, get_config, DEVICE
from model import Model
import torch
import os
# from preprocess import process_wav_to_tokens

args = get_args()

tokenizer = AutoTokenizer.from_pretrained("../data/Haptic/pretrain/updated_llama_tokenizer_steps_binning.pt")
# tokenizer = AutoTokenizer.from_pretrained("../data/Haptic/pretrain/updated_llama_tokenizer_encodec.pt")


def build_dpo_dataset_from_ranking(json_dir, category):

    dpo_category_data = []
    dpo_data = load_json(json_dir)
    for item in dpo_data:
        dpo_category_data.extend(item[category])

    return dpo_category_data
         

def load_model(epoch, device, output_dir, version, mode, model_file_url=None):
        if model_file_url is None or len(model_file_url) == 0:
            model_file_url = os.path.join(output_dir, "full_lora_{}_{}.pth".format(epoch, version))
        if os.path.exists(model_file_url):
            model = Model(args, mode=mode)

            lora_state_dict = torch.load(model_file_url)
            state_name, model_name = [], []
            # print('--------------------original**model-------------------')
            for name, param in model.named_parameters():
                model_name.append(name)
            # print('--------------------state**dict-------------------')
            for name in lora_state_dict.keys():
                state_name.append(name)
            print('len1:{},len2:{}'.format(len(model_name), len(state_name)))
            missing_keys, unexpected_keys = model.load_state_dict(lora_state_dict, strict=False)
            # print("Missing keys:", missing_keys)
            # print("Unexpected keys:", unexpected_keys)
            model.to(device)
        else:
            model = None
        return model

def dpo_train(json_dir, epoch, model_dir, model_url, version, category):

    dpo_data = build_dpo_dataset_from_ranking(json_dir,category)
    train_dataset = Dataset.from_list(dpo_data)


    model = load_model(epoch=epoch, device=DEVICE, output_dir=model_dir, version=version, mode='encodec', model_file_url=model_url)
    breakpoint()

    config = DPOConfig(
        output_dir="./dpo-{}-{}-caption-model".format(epoch, category)
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model()

json_dir = r'../data/Haptic/jsons/dpo_data.json'
category_list = ['sen', 'emo', 'ass']
epoch = 2
model_dir = r'.'
model_url = r'./pretrained_model/encodec/encodec_full_lora_2_5.1.pth'
version = '6.20'
for category in category_list:
    dpo_train(json_dir, epoch, model_dir, model_url, version=version, category=category)


