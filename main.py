import iter
import nets_no_mask
import nets
import nets_batch_first
import training
import argparse
import opts
import utils
import os
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
                             ftest=opt.ftest,
                             bsz=opt.bsz,
                             device=opt.gpu,
                             pretrain=opt.pretrain,
                             min_freq=opt.min_freq)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    SEQ = iters['SEQ']

    embedding = nn.Embedding(num_embeddings=len(SEQ.vocab.itos),
                             embedding_dim=opt.edim,
                             padding_idx=SEQ.vocab.stoi[PAD])

    model = None
    if opt.mask == 'no_mask':
        encoder = nets_no_mask.StructLSTM(opt.edim,
                                  opt.hdim,
                                  opt.sema_dim,
                                  opt.stru_dim,
                                  opt.dropout,
                                  SEQ.vocab.stoi[PAD])

        model = nets_no_mask.StructNLI(encoder, embedding, opt.dropout).to(device)

    if opt.mask == 'mask':
        # encoder = nets.StructLSTM(opt.edim,
        #                                   opt.hdim,
        #                                   opt.sema_dim,
        #                                   opt.stru_dim,
        #                                   opt.dropout,
        #                                   SEQ.vocab.stoi[PAD])
        #
        # model = nets.StructNLI(encoder, embedding, opt.dropout).to(device)
        encoder = nets_batch_first.StructLSTM(opt.edim,
                                              opt.hdim,
                                              opt.dropout,
                                              SEQ.vocab.stoi[PAD])

        model = nets_batch_first.StructNLI(encoder, embedding, opt.dropout).to(device)


    # utils.init_model_normal(model)
    utils.init_model_xavier(model)

    if opt.pretrain:
        # model.embedding.weight.data.copy_(SEQ.vocab.vectors)
        utils.load_pretrain(embedding, SEQ.vocab.vectors)

    if opt.fix_emb:
        embedding.weight.requires_grad = False

    if opt.fload is not None:
        model_fname = opt.fload
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_path = os.path.join(RES, model_fname)
        model_dict = torch.load(model_path, map_location=location)
        model.load_state_dict(model_dict)
        print('Loaded from ' + model_path)

    criterion = nn.CrossEntropyLoss()

    optimizer = None
    if opt.optim == 'adagrad':
        optimizer = optim.Adagrad(params=filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.lr,
                                  weight_decay=opt.wdecay)

    if opt.optim == 'adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                               lr=opt.lr,
                               weight_decay=opt.wdecay)

    param_str = utils.param_str(opt)
    for key, val in param_str.items():
        print(str(key) + ': ' + str(val))

    training.train(model, iters, opt, criterion, optimizer)

