import torch.nn as nn
import torch
import torch.cuda
from torch.nn import functional as F
import crash_on_ipy
import numpy as np
import utils

class MatrixTree(nn.Module):
    """Implementation of the matrix-tree theorem for computing marginals
    of non-projective dependency parsing. This attention layer is used
    in the paper "Learning Structured Text Representations."


    :cite:`DBLP:journals/corr/LiuL17d`
    """
    def __init__(self, eps=1e-5):
        self.eps = eps
        super(MatrixTree, self).__init__()

    def forward(self, A, r):
        # A: (bsz, seq_len, seq_len)
        # r: (bsz, seq_len, 1)
        bsz, seq_len = A.shape[0], A.shape[1]
        L = -A
        L[:, range(seq_len), range(seq_len)] = A.sum(dim=1)
        LL = L[:, 1:, :]
        LL = torch.cat([r.squeeze(-1).unsqueeze(1), LL], dim=1)

        LL_inv = utils.inv(LL)
        d0 = (r.squeeze(-1) * LL_inv[:, :, 0]).unsqueeze(1)
        # LL_inv_diag: (bsz, seq_len, 1)
        LL_inv_diag = LL_inv[:, range(seq_len), range(seq_len)].unsqueeze(-1)
        term1 = (A.transpose(1, 2) * LL_inv_diag).transpose(1, 2)
        term2 = A * LL_inv.transpose(1, 2)
        term1[:, :, 0] = 0
        term2[:, 0, :] = 0
        d = term1 - term2
        d = torch.cat([d0, d], dim=1)

        return d


class StructLSTM(nn.Module):

    def __init__(self,
                 edim,
                 hdim,
                 sema_dim,
                 stru_dim,
                 dropout,
                 padding_idx):

        super(StructLSTM, self).__init__()
        self.edim = edim
        self.hdim = hdim
        self.sema_dim = sema_dim
        self.stru_dim = stru_dim
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.bilstm = nn.LSTM(input_size=edim,
                              hidden_size=hdim//2,
                              dropout=dropout,
                              bidirectional=True)
        self.struct_atten = MatrixTree()
        self.d2tp = nn.Linear(stru_dim, stru_dim)
        self.d2tc = nn.Linear(stru_dim, stru_dim)
        self.w_a = nn.Parameter(torch.randn(stru_dim, stru_dim))
        self.w_root = nn.Linear(stru_dim, 1, bias=False)
        self.root_emb = nn.Parameter(torch.randn(sema_dim))
        self.w_r = nn.Linear(sema_dim * 2, sema_dim)
        self.eps = 1e-5

    def forward(self, input):
        print('--' * 10 + 'enc' + '--' * 10)

        # output: (seq_len, bsz, hdim)
        output, h = self.bilstm(input)

        output_forward = output[:,:,:self.hdim//2]
        output_backward = output[:,:,self.hdim//2:]

        # vec_sema: (bsz, seq_len, sema_dim)
        # vec_stru: (bsz, seq_len, stru_dim)
        vec_sema = torch.cat([output_forward[:,:,:self.sema_dim//2],
                             output_backward[:,:,:self.sema_dim//2]], dim=-1)
        vec_stru = torch.cat([output_forward[:,:,self.sema_dim//2:],
                             output_backward[:,:,self.sema_dim//2:]], dim=-1)
        vec_sema = vec_sema.transpose(0, 1)
        vec_stru = vec_stru.transpose(0, 1)
        print('1:', vec_sema)
        print('1:', vec_stru)

        # tp/tc: (bsz, seq_len, stru_dim)
        tp = F.tanh(self.d2tp(vec_stru))
        tc = F.tanh(self.d2tc(vec_stru))
        print('2:', tp)
        print('2:', tc)

        # f: (bsz, seq_len, seq_len)
        # f_r: (bsz, seq_len, 1)
        f = tp.matmul(self.w_a). \
            matmul(tc.transpose(1, 2))
        f_r = self.w_root(vec_stru)
        print('3:', f)
        print('3:', f_r)

        f = f.exp()
        f_r = f_r.exp()
        seq_len = f.shape[1]
        f[:, range(seq_len), range(seq_len)] = 0

        print('4:', f)
        print('4:', f_r)
        # a: (bsz, seq_len + 1, seq_len)
        a = self.struct_atten(f + self.eps, f_r + self.eps)
        print('4:', a)

        # a_p: (bsz, seq_len, seq_len + 1)
        a_p = a.transpose(1, 2)

        root_embs = self.root_emb.expand_as(vec_sema[:, :1, :])
        # vec_sem_root: (bsz, seq_len + 1, sema_dim)
        vec_sem_root = torch.cat([root_embs, vec_sema], dim=1)
        print('5:', vec_sem_root)

        # p: (bsz, seq_len, sema_dim)
        p = torch.matmul(a_p, vec_sem_root)
        # output: (seq_len, bsz, sema_dim)
        output = F.leaky_relu(self.w_r(torch.cat([vec_sema, p], dim=-1)))
        output = output.transpose(0, 1)
        print('6:', output)

        return output

class InterAttention(nn.Module):
    def __init__(self, sema_dim):

        super(InterAttention, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(sema_dim, sema_dim),
                                 nn.ReLU(),
                                 nn.Linear(sema_dim, sema_dim))
        self.inf = -1e10

    def forward(self, r1, r2):
        print('--' * 10 + 'atten' + '--' * 10)

        # r: (bsz, seq_len, sema_dim)
        r1 = self.mlp(r1).transpose(0, 1)
        r2 = self.mlp(r2).transpose(0, 1)
        print('1:', r1)
        print('1:', r2)

        print('2:', r1)
        print('2:', r2)

        # o: (bsz, seq_len1, seq_len2)
        o = torch.matmul(r1, r2.transpose(1, 2))
        print('3:', o)
        o1 = F.softmax(o, dim=2)
        o2 = F.softmax(o, dim=1)
        print('o1:', o1)
        print('o2:', o2)

        # r_c: (bsz, seq_len, sema_dim)
        r1_c = torch.matmul(o1, r2)
        r2_c = torch.matmul(o2.transpose(1, 2), r1)
        print('4:', r1_c)
        print('4:', r2_c)

        # r_pooling: (bsz, sema_dim)
        # for the processing above, no mask needed for these two steps
        r1_pooling = torch.cat([r1, r1_c], dim=-1).sum(1) / r1.shape[1]
        r2_pooling = torch.cat([r2, r2_c], dim=-1).sum(1) / r2.shape[1]
        print('5:', r1_pooling)
        print('5:', r2_pooling)

        return r1_pooling, r2_pooling

class StructNLI(nn.Module):

    def __init__(self, encoder, embedding, dropout):

        super(StructNLI, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.padding_idx = encoder.padding_idx
        sema_dim = encoder.sema_dim
        self.mlp = nn.Sequential(nn.Linear(sema_dim * 4, sema_dim),
                                 nn.ReLU(),
                                 nn.Linear(sema_dim, 3))
        self.inter_atten = InterAttention(sema_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1, seq2):
        print('seq1:', seq1)
        print('seq2:', seq2)
        embs1 = self.embedding(seq1)
        embs2 = self.embedding(seq2)

        embs1 = self.dropout(embs1)
        embs2 = self.dropout(embs2)

        # r: (seq_len, bsz, sema_dim)
        r1 = self.encoder(embs1)
        r2 = self.encoder(embs2)
        print('r1:', r1)
        print('r2:', r2)

        # mask for inter_atten...
        # r_pooling: (bsz, sema_dim * 2)
        r1_pooling, r2_pooling = self.inter_atten(r1, r2)
        r = torch.cat([r1_pooling, r2_pooling], dim=-1)
        print('r:', r)

        # output: (bsz, 3)
        output = self.mlp(r)
        print('output:', output)
        exit()
        return output

if __name__ == "__main__":
    dtree = MatrixTree()
    A = torch.randint(0, 9, (1, 5, 5))
    A[:, range(5), range(5)] = 0
    r = torch.randint(0, 9, (1, 5, 1))
    print(A)
    print(r)
    marg = dtree.forward(A, r)
    print(marg)
    # print(marg.shape)
    # print(marg.sum(1))