import argparse
import os
from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=300)
    group.add_argument('-hdim', type=int, default=300)
    group.add_argument('-sema_dim', type=int, default=200)
    group.add_argument('-stru_dim', type=int, default=100)
    group.add_argument('-dropout', type=float, default=0.2)
    group.add_argument('-fix_emb', default=True, action='store_true')
    group.add_argument('-inter_atten', default=True, action='store_true')
    group.add_argument('-pooling', type=str, default='mean') # mean, max, self_attention
    group.add_argument('-clf', type=str, default='cat') # cat, 4way

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-min_freq', type=float, default=5)
    group.add_argument('-seed', type=int, default=1000)
    group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-lr', type=float, default=1e-4)
    group.add_argument('-ftrain', type=str, default=os.path.join(DATA, 'snli_1.0_train.txt'))
    group.add_argument('-fvalid', type=str, default=os.path.join(DATA, 'snli_1.0_dev.txt'))
    group.add_argument('-ftest', type=str, default=os.path.join(DATA, 'snli_1.0_test.txt'))
    group.add_argument('-fload', type=str, default=None)
    group.add_argument('-pretrain', type=str, default='glove.840B.300d')
    group.add_argument('-wdecay', type=float, default=1e-4)
    group.add_argument('-optim', type=str, default='adam')


