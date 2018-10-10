from macros import *
from . import snli_binary_tree, snli_struct_atten, snli_cslstm

def general_opts(parser):
    group = parser.add_argument_group('general')
    group.add_argument('-enc', type=str, default='cslstm')
    # group.add_argument('-enc', type=str, default='binary_tree')
    # group.add_argument('-enc', type=str, default='struct_atten')

    group.add_argument('-task', type=str, default='snli')

    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-nepoch', type=int, default=100)
    group.add_argument('-save_per', type=int, default=1)

def select_opt(opt, parser):
    if opt.task == 'snli' and opt.enc == 'binary_tree':
        snli_binary_tree.model_opts(parser)
        snli_binary_tree.train_opts(parser)
    elif opt.task == 'snli' and opt.enc == 'struct_atten':
        snli_struct_atten.model_opts(parser)
        snli_struct_atten.train_opts(parser)
    elif opt.task == 'snli' and opt.enc == 'cslstm':
        snli_cslstm.model_opts(parser)
        snli_cslstm.train_opts(parser)
    else:
        raise ModuleNotFoundError

    return parser

