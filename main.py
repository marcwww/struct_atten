import iter
import nets
import training
import argparse
import opts
import utils
import torch
from torch import nn
from torch import optim
from macros import *
import crash_on_ipy


if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    utils.init_seed(opt.seed)

    iters = iter.build_iters(ftrain=opt.ftrain,
                     fvalid=opt.fvalid,
                     bsz=opt.bsz,
                     device=opt.gpu)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    SEQ = iters['SEQ']

    embedding = nn.Embedding(num_embeddings=len(SEQ.vocab.itos),
                             embedding_dim=opt.edim,
                             padding_idx=SEQ.vocab.stoi[PAD])

    encoder = nets.StructLSTM(opt.edim,
                              opt.hdim,
                              opt.sema_dim,
                              opt.stru_dim,
                              opt.dropout,
                              SEQ.vocab.stoi[PAD])

    model = nets.StructNLI(encoder, embedding).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adagrad(params=filter(lambda p: p.requires_grad, model.parameters()),
    #                        lr=opt.lr)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=opt.lr)

    param_str = utils.param_str(opt)
    for key, val in param_str.items():
        print(str(key) + ': ' + str(val))

    training.train(model, iters, opt, criterion, optimizer)

