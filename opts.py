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

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)
    group.add_argument('-bsz', type=int, default=32)
    # group.add_argument('-lr', type=float, default=5e-2)
    # group.add_argument('-lr', type=float, default=2e-4)
    group.add_argument('-lr', type=float, default=1e-4)
    group.add_argument('-ftrain', type=str, default=os.path.join(DATA, 'snli_1.0_train.txt'))
    group.add_argument('-fvalid', type=str, default=os.path.join(DATA, 'snli_1.0_dev.txt'))
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-nepoch', type=int, default=10)
    group.add_argument('-save_per', type=int, default=1)


