import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import logging
import  os
import math
from config import get_args, get_config
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


logger = logging.getLogger(__name__)
args = get_args()
class AdapterConfig:
    project_hidden_size: int = args.hidden_size
    hidden_act: str = "gelu"
    adapter_size: int = 64  # 64
    adapter_initializer_range: float = 0.001
    is_decoder: bool = False
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 544
    out_hidden_size: int = project_hidden_size
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    max_position_embeddings: int = 514
    num_attention_heads: int = 12
    num_labels: int = 2
    output_attentions: bool = False
    output_hidden_states: bool = False
    torchscript: bool = False
    type_vocab_size: int = 1
    vocab_size: int = 50265

class FFN_Adapter(nn.Module):
    def __init__(self, args):
        super(FFN_Adapter, self).__init__()
        self.adapter_config =  AdapterConfig()
        self.multi = args.multi
        self.adapter_layer = args.adapter_layer

        in_dim = self.adapter_config.project_hidden_size

        self.adapter_down_project = nn.Linear(in_dim,self.adapter_config.adapter_size)
        self.adapter_up_project = nn.Linear(self.adapter_config.adapter_size,in_dim)
        self.adapter_down_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                                   size=(self.adapter_config.adapter_size, in_dim,)))
        self.adapter_down_project.bias = torch.nn.Parameter(torch.zeros(self.adapter_config.adapter_size))

        self.adapter_up_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                    size=(in_dim, self.adapter_config.adapter_size,)))
        self.adapter_up_project.bias = torch.nn.Parameter(torch.zeros(in_dim))
        self.adapter_linear = nn.Linear(in_dim,self.adapter_config.out_hidden_size)

    def forward(self, hidden_states, visualize=True, epoch=10):
            
        down_output = self.adapter_down_project(hidden_states)
        down_output_nolinear = F.sigmoid(down_output)
        up_output = self.adapter_up_project(down_output_nolinear)
        output = up_output + hidden_states
        output = self.adapter_linear(output)
            
        if visualize:
            pool_hidden_state = torch.mean(hidden_states,dim=1).cpu().detach().numpy()
            
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pool_hidden_state)
            X_pca = PCA(n_components=2).fit_transform(pool_hidden_state)

            ckpt_dir="images"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=None,label="t-SNE")
            plt.legend()
            plt.subplot(122)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=None,label="PCA")
            plt.legend()
            plt.savefig('images/orig_tsne-pca_{}.png'.format(str(epoch)), dpi=120)
            
            pool_fusion = torch.mean(output, dim=1).cpu().detach().numpy()
            
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pool_fusion)
            X_pca = PCA(n_components=2).fit_transform(pool_fusion)


            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=None,label="t-SNE")
            plt.legend()
            plt.subplot(122)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=None,label="PCA")
            plt.legend()
            plt.savefig('images/adapter_tsne-pca_{}.png'.format(str(epoch), dpi=120))
            
        return output


    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

