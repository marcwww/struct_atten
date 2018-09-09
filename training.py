import utils
import os
from macros import *
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score
import torch
import numpy as np

def valid(model, valid_iter):
    pred_lst = []
    true_lst = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            seq1, seq2, lbl = batch.seq1, batch.seq2, batch.lbl
            output= model(seq1, seq2)
            pred = output.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accuracy = accuracy_score(true_lst, pred_lst)

    return accuracy

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']

    basename = "{}-{}".format('struct_atten', utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(RES, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')

    losses = []
    best_performance = 0
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            seq1, seq2, lbl = batch.seq1, batch.seq2, batch.lbl

            model.train()
            model.zero_grad()
            output = model(seq1, seq2)
            if i == 2:
                exit()

            loss = criterion(output.view(-1, len(LBL)), lbl)
            losses.append(loss.item())
            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()

            loss = {'clf_loss': loss.item()}

            utils.progress_bar(i / len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                accurracy = \
                    valid(model, valid_iter)
                loss_ave = np.array(losses).sum() / len(losses)
                losses = []
                log_str = '{\'Epoch\':%d, \'Format\':\'a/l\', \'Metrics\':[%.4f, %.4f]}' % \
                          (epoch, accurracy, loss_ave)
                print(log_str)
                with open(log_path, 'a+') as f:
                    f.write(log_str + '\n')

                if accurracy > best_performance:
                    best_performance = accurracy
                    model_fname = basename + ".model"
                    save_path = os.path.join(RES, model_fname)
                    print('Saving to ' + save_path)
                    torch.save(model.state_dict(), save_path)
