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
    # group.add_argument('-fix_emb', default=False, action='store_true')
    # group.add_argument('-dropout', type=float, default=0.3)
    # group.add_argument('-dropout', type=float, default=0)
    group.add_argument('-enc', type=str, default='binary_tree')
    # group.add_argument('-enc', type=str, default='struct_atten')
    group.add_argument('-inter_atten', default=False, action='store_true')
    group.add_argument('-pooling', type=str, default='self_attention') # mean, max, self_attention
    group.add_argument('-clf', type=str, default='4way') # cat, 4way

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-min_freq', type=float, default=5)
    group.add_argument('-seed', type=int, default=1000)
    # group.add_argument('-bsz', type=int, default=128)
    group.add_argument('-bsz', type=int, default=64)
    # group.add_argument('-bsz', type=int, default=2)
    # group.add_argument('-bsz', type=int, default=32)
    # group.add_argument('-lr', type=float, default=1e-5)
    group.add_argument('-lr', type=float, default=2e-4)
    # group.add_argument('-lr', type=float, default=2e-5)
    # group.add_argument('-lr', type=float, default=4e-4)
    # group.add_argument('-IAV', type=float, default=0.1)
    # group.add_argument('-lr', type=float, default=1e-5)
    # group.add_argument('-lr', type=float, default=0.05)
    group.add_argument('-ftrain', type=str, default=os.path.join(DATA, 'snli_1.0_train.txt'))
    group.add_argument('-fvalid', type=str, default=os.path.join(DATA, 'snli_1.0_dev.txt'))
    group.add_argument('-ftest', type=str, default=os.path.join(DATA, 'snli_1.0_test.txt'))
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-nepoch', type=int, default=100)
    group.add_argument('-save_per', type=int, default=1)
    group.add_argument('-fload', type=str, default=None)
    # group.add_argument('-fload', type=str, default='struct_atten-1538545731.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1538186850.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1538216307.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1537290454.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1536370010.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1536483472.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1536483789.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1536844414.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1536591069.model')
    # group.add_argument('-fload', type=str, default='struct_atten-1536483451.model')
    group.add_argument('-pretrain', type=str, default='glove.840B.300d')
    # group.add_argument('-pretrain', type=str, default=None)
    group.add_argument('-wdecay', type=float, default=1e-4)
    # group.add_argument('-optim', type=str, default='adagrad')
    group.add_argument('-optim', type=str, default='adam')


