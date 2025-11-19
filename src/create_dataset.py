import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call
from datasets import Dataset, Audio
from config import Config
from config import get_args

import torch
import torch.nn as nn
import json as js
from utils.util import load_pickle, to_pickle, load_csv, load_json

args = get_args()

class HAPTIC:
    def __init__(self, config):

        
        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        ##v1, full prompt for category + full haptic token
        ##v2, short prompt for category + full haptic token
        ##v3, short prompt for category + short haptic token
        ##v4, full prompt for category + short haptic token

        # self.train = load_pickle(DATA_PATH + '/updated/train_short_haptic.pkl')
        # self.dev = load_pickle(DATA_PATH + '/updated/valid_short_haptic.pkl')
        # self.test = load_pickle(DATA_PATH + '/updated/test_short_haptic.pkl')

        self.train = load_pickle(DATA_PATH + '/updated/train_{}_haptic_5_1{}.pkl'.format(args.mode, args.suffix))
        self.dev = load_pickle(DATA_PATH + '/updated/valid_{}_haptic_5_1{}.pkl'.format(args.mode, args.suffix))
        self.test = load_pickle(DATA_PATH + '/updated/test_{}_haptic_5_1{}.pkl'.format(args.mode, args.suffix))



        ###just for separate test generation with the newly generated haptic###
        # self.test = load_pickle(DATA_PATH + '/updated/test_step.pkl')
        ### this test set is just for caption generation for human verification without ground caption
        ###just for separate test generation with the newly generated haptic###

    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.dev
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


##load_data test
# test = Config()
# haptic = Haptics()


