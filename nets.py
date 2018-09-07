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


    def forward(self, input):
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

        # tp/tc: (bsz, seq_len, stru_dim)
        tp = F.tanh(self.d2tp(vec_stru))
        tc = F.tanh(self.d2tc(vec_stru))

        # f: (bsz, seq_len, seq_len)
        # f_r: (bsz, seq_len, 1)
        f = tp.matmul(self.w_a). \
            matmul(tc.transpose(1, 2))
        f_r = self.w_root(vec_stru)

        seq_len = f.shape[1]
        f[:, range(seq_len), range(seq_len)] = 0
        f = f.exp()
        f_r = f_r.exp()

        # a: (bsz, seq_len + 1, seq_len)
        a = self.struct_atten(f, f_r)

        # a = torch.cat([f_r, f], dim=-1).transpose(1, 2)
        # a_p: (bsz, seq_len, seq_len + 1)
        a_p = a.transpose(1, 2)

        root_embs = self.root_emb.expand_as(vec_sema[:, :1, :])
        # vec_sem_root: (bsz, seq_len + 1, sema_dim)
        vec_sem_root = torch.cat([root_embs, vec_sema], dim=1)

        # p: (bsz, seq_len, sema_dim)
        p = torch.matmul(a_p, vec_sem_root)
        # output: (seq_len, bsz, sema_dim)
        output = F.leaky_relu(self.w_r(torch.cat([vec_sema, p], dim=-1)))
        output = output.transpose(0, 1)

        return output

class InterAttention(nn.Module):
    def __init__(self, sema_dim):

        super(InterAttention, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(sema_dim, sema_dim),
                                 nn.ReLU(),
                                 nn.Linear(sema_dim, sema_dim))

    def forward(self, r1, r2):
        # r: (bsz, seq_len, sema_dim)
        r1 = self.mlp(r1).transpose(0, 1)
        r2 = self.mlp(r2).transpose(0, 1)

        # o: (bsz, seq_len1, seq_len2)
        o = torch.matmul(r1, r2.transpose(1, 2))
        o1 = F.softmax(o, dim=1)
        o2 = F.softmax(o, dim=0)

        # r_c: (bsz, seq_len, sema_dim)
        r1_c = torch.matmul(o1, r2)
        r2_c = torch.matmul(o2.transpose(1, 2), r1)

        # r_pooling: (bsz, sema_dim)
        r1_pooling = torch.cat([r1, r1_c], dim=-1).sum(1)/r1.shape[1]
        r2_pooling = torch.cat([r2, r2_c], dim=-1).sum(1)/r2.shape[1]

        return r1_pooling, r2_pooling

class StructNLI(nn.Module):

    def __init__(self, encoder, embedding):

        super(StructNLI, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        sema_dim = encoder.sema_dim
        self.mlp = nn.Sequential(nn.Linear(sema_dim * 4, sema_dim),
                                 nn.ReLU(),
                                 nn.Linear(sema_dim, 3))
        self.inter_atten = InterAttention(sema_dim)

    def forward(self, seq1, seq2):
        embs1 = self.embedding(seq1)
        embs2 = self.embedding(seq2)

        # r: (seq_len, bsz, sema_dim)
        r1 = self.encoder(embs1)
        r2 = self.encoder(embs2)

        # r_pooling: (bsz, sema_dim * 2)
        r1_pooling, r2_pooling = self.inter_atten(r1, r2)
        r = torch.cat([r1_pooling, r2_pooling], dim=-1)

        # output: (bsz, 3)
        output = self.mlp(r)

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