# -*- encoding:utf-8 -*-
import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
import random
import warnings
import json as js
from transformers import AutoProcessor, AutoTokenizer
from transformers import ASTModel, ASTConfig, AutoModel,EncodecModel,EncodecConfig,Wav2Vec2Config, Wav2Vec2Model

class Feature_Extractor(nn.Module):
    def __init__(self, extractor_name='AST',
                  sample_rate=16000):
        
        super().__init__()
        self.extractor_name = extractor_name
        self.sample_rate = sample_rate

        if extractor_name == 'AST':
            self.extractor = AutoProcessor\
            .from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            # configuration = ASTConfig()
            # self.encoder = ASTModel(configuration)
            self.audio_encoder = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        if extractor_name == 'Wav2Vec':
            self.extractor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        if extractor_name == 'EncoDec':
            self.extractor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
            self.audio_encoder = EncodecModel.from_pretrained("facebook/encodec_24khz")

        
    def forward(self, sample):

        input_values = self.extractor(sample, sampling_rate=self.sample_rate, return_tensors="pt")
        if self.extractor_name in ['AST', 'Wav2Vec']:
            haptic_features = self.audio_encoder(input_values['input_values']).last_hidden_state
        else:
            haptic_features = self.audio_encoder(input_values["input_values"],
                                                  input_values["padding_mask"]).audio_values


        print('haptic_features.shape:{}'.format(haptic_features.shape))

        return haptic_features.transpose(1, 2)
    