# -*- encoding:utf-8 -*-

import pickle
import json as js
import torch
import torch.nn as nn
import threading
from torch._utils import ExceptionWrapper
import logging
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine

from collections import defaultdict


from pickle import load
from matplotlib.pylab import plt
from numpy import arange

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

name = 'princeton-nlp/sup-simcse-bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(DEVICE)
 
def plot_loss(train_pkl, valid_pkl, epoch_num):
    # Load the training and validation loss dictionaries
    train_loss = load(open(train_pkl, 'rb'))
    val_loss = load(open(valid_pkl, 'rb'))
    
    # Retrieve each dictionary's values
    train_values = train_loss.values()
    val_values = val_loss.values()
    
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, epoch_num+1)
    # print(val_values)
    
    # Plot and label the training and validation loss values
    plt.plot(epochs, train_values, label='Training Loss')
    # plt.plot(epochs, val_values, label='Validation Loss')
    
    # Add in a title and axes labels
    # plt.title('Training and Validation Loss')
    # plt.title('Validation Loss')
    plt.title('Train Loss')


    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Set the tick locations
    plt.xticks(arange(0, epoch_num+1, 1))
    
    # Display the plot
    plt.legend(loc='best')
    plt.savefig('./train_loss.png',dpi=900)

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_json(json_path):
    with open(json_path) as f:
        data = js.load(f)

    return data

def to_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        js.dump(data, f, ensure_ascii=False, indent=4)
    

def calculate_average_metric(bleu1_list, bleu4_list, rouge_meteor_list, category_list):

    def average(value_list):

        value_array = np.array(value_list)

        avg_bleu1 = np.mean(value_array, axis=0)
        avg_max, avg_min = avg_bleu1[0], avg_bleu1[1]

        return avg_max, avg_min


    def pairwise(value, category):

        category_dict = {}

        for val, cat in zip(value, category):
            if cat not in category_dict:
                category_dict[cat] = []
                category_dict[cat].append(val)
            else:
                category_dict[cat].append(val)
            
        if len(value[0]) == 2:
            category_mean_max = {cat: np.mean(np.array(vals),axis=0)[0] for cat, vals in category_dict.items()}
            category_mean_min = {cat: np.mean(np.array(vals),axis=0)[1] for cat, vals in category_dict.items()}

            return category_mean_min, category_mean_max
        else:

            category_mean = {cat: sum(vals) / len(vals) for cat, vals in category_dict.items()}
            return category_mean
    
    category_avg_bleu1 = pairwise(bleu1_list,category_list)
    category_avg_bleu4 = pairwise(bleu4_list,category_list)
    category_avg_rouge_meteor = pairwise(rouge_meteor_list, category_list)

    avg_bleu1 = average(bleu1_list)
    avg_bleu4 = average(bleu4_list)
    rouge_score, meteor_score = average(rouge_meteor_list) 

    return avg_bleu1, avg_bleu4, rouge_score, meteor_score, category_avg_bleu1,category_avg_bleu4,category_avg_rouge_meteor 

def calculate_similarity(generated_caption_list):
    inputs_caption = tokenizer(generated_caption_list, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        embeddings_caption = model(**inputs_caption, output_hidden_states=False, return_dict=True).pooler_output
    
    des_len = len(embeddings_caption)
    embeddings_caption = embeddings_caption.cpu()
    cosine_caption = np.zeros([des_len, des_len])
    for i in range(des_len):
        for j in range(des_len):
            cosine_sim_i_j = 1 - cosine(embeddings_caption[i], embeddings_caption[j])
            cosine_caption[i][j] = round(cosine_sim_i_j, 2)

    global_similarity = np.mean(cosine_caption)
    row_similarity = np.mean(cosine_caption, axis=0)
    return global_similarity, row_similarity, cosine_caption

def read_csv(path):
    with open(path, 'rb') as f:
        return f.readlines()
    
def load_csv(path):
    data = pd.read_csv(path, sep=',', header=0)
    return data

def load_json(path):
    with open(path, 'rb') as f:
        data = js.load(f)
    return data

def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(fct, model, inputs, device_ids):
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                   for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger


# train_pkl = r'../train_loss_1.pkl'
# valid_pkl = r'../val_loss_1.pkl'

# plot_loss(train_pkl, valid_pkl, 20)