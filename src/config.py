import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import torch


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('data/Haptic/pkl')
data_dict = {'haptic': data_dir.joinpath('Haptic')} 
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}
tokenizer_dict = {}


def get_args():
    parser = argparse.ArgumentParser(description='Haptic Caption')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='haptic', choices=['haptic'],
                        help='dataset to use (default: haptic)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')
    parser.add_argument('--ground_data', type=str, default='../data/Haptic/ground_data.json',
                        help='path for storing the ground data')
    parser.add_argument('--mode', type=str, default='steps_binning', choices=['steps_binning', 'encodec'],
                        help='tokenizer type')
    parser.add_argument('--suffix', type=str, default='_aug', choices=['', '_aug'],
                        help='suffix')
    
    # Modules:

    parser.add_argument('--param_mode', type=str, default='lora', choices=['lora', 'adapter'],
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--dropout_t', type=float, default=0.2,
                        help='dropout of text representation')
    parser.add_argument('--dropout_h', type=float, default=0.5,
                        help='dropout of haptic representation')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')

    # Activations
    parser.add_argument('--hidden_size', default=768)
    parser.add_argument('--gradient_accumulation_step', default=5)

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=0.0001,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--learn_rate', type=float, default=0.0001,
                        help='initial learning rate for main model parameters (default: 1e-3)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='gradient_accumulation_steps')


    #### subnetwork parameter
    parser.add_argument('--embed_dropout', type=float, default=1e-4,
                        help='embed_drop')
    parser.add_argument('--attn_dropout', type=float, default=1e-4,
                        help='attn_dropout')
    parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='num_heads')
    parser.add_argument('--relu_dropout', type=float, default=1e-4,
                        help='relu_dropout')
    parser.add_argument('--res_dropout', type=float, default=1e-4,
                        help='res_dropout')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    #### subnetwork parameter
        
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./')
    
    
    ### 可视化
    parser.add_argument('--visualize', type=bool, default=False)
    ### 可视化

    args = parser.parse_args()
    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.mode = mode
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='haptic', mode='train', batch_size=2):
    config = Config(data=dataset, mode=mode)
    
    config.dataset = dataset
    config.batch_size = batch_size

    return config