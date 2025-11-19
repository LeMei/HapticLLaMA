import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from solver import Solver
# from solver_test import Solver
# from solver_test_sep import Solver
from config import get_args, get_config
# from dataloader_test import get_loader
from dataloader import get_loader

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())
    print('current dataset:-----------{}---------'.format(dataset))
    print('current mode:-----------{}-----------'.format(args.mode))
    aug = False
    if len(args.suffix)>1:
        aug = True
    print('aug mode:-----------{}-------------'.format(aug))
    bs = args.batch_size
    
    set_seed(args.seed)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size)
    
    train_loader = get_loader(args, train_config, shuffle=True)
    print('{} training data loaded!'.format(args.n_train))

    valid_loader = get_loader(args, valid_config, shuffle=False)
    print('{} valid data loaded!'.format(args.n_valid))

    # architecture parameters
    args.dataset = args.data = dataset
    args.when = args.when


    test_config = get_config(dataset, mode='test', batch_size=args.batch_size)

    test_loader = get_loader(args, test_config, shuffle=False)

    print('{} {} test data loaded!'.format(args.n_test, args.dataset))
    print('Finish loading the data....')

    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    
    solver.train_and_eval()
