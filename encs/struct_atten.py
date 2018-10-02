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
    def __init__(self, eps=1):
        self.eps = eps
        super(MatrixTree, self).__init__()

    def forward(self, A, r):
        # A: (bsz, seq_len, seq_len)
        # r: (bsz, seq_len, 1)
        bsz, seq_len = A.shape[0], A.shape[1]
        indices_n = list(range(seq_len))

        L = -A
        L[:, indices_n, indices_n] = A.sum(dim=1)
        LL = L[:, 1:, :]
        LL = torch.cat([r.squeeze(-1).unsqueeze(1), LL], dim=1)

        LL_diag = LL[:, indices_n, indices_n]
        LL[:, indices_n, indices_n] = LL_diag.\
            masked_fill_(LL_diag.eq(0), self.eps)
        LL_inv = utils.inv4(LL)

        assert np.isnan(LL_inv.view(-1).data).sum().item() == 0

        d0 = (r.squeeze(-1) * LL_inv[:, :, 0]).unsqueeze(1)
        # LL_inv_diag: (bsz, seq_len, 1)
        LL_inv_diag = LL_inv[:, range(seq_len), range(seq_len)].unsqueeze(-1)
        term1 = (A.transpose(1, 2) * LL_inv_diag).transpose(1, 2)
        term2 = A * LL_inv.transpose(1, 2)
        term1[:, :, 0] = 0
        term2[:, 0, :] = 0
        d = term1 - term2

        return d0, d

class StructLSTM(nn.Module):

    def __init__(self, word_dim, hidden_dim, dropout, padding_idx, bidirectional=True):
        super(StructLSTM, self).__init__()
        self.edim = word_dim
        self.sema_dim = hidden_dim // 3 * 2
        self.stru_dim = hidden_dim // 3
        self.dropout = dropout
        self.hidden_dim = hidden_dim // 2
        self.padding_idx = padding_idx
        self.bilstm = nn.LSTM(input_size=word_dim, hidden_size=hidden_dim // 2,
                              dropout=dropout, bidirectional=bidirectional)
        self.struct_atten = MatrixTree()

        self.d2tp = nn.Linear(self.stru_dim, self.stru_dim)
        self.d2tc = nn.Linear(self.stru_dim, self.stru_dim)
        self.w_a = nn.Parameter(torch.randn(self.stru_dim, self.stru_dim))
        self.w_root = nn.Linear(self.stru_dim, 1, bias=False)
        self.root_emb = nn.Parameter(torch.randn(self.sema_dim))
        # self.w_r = nn.Linear(self.sema_dim * 3, hidden_dim)
        self.w_r = nn.Linear(self.sema_dim * 3, self.sema_dim)
        self.eps = 1e-5
        self.inf = -1e10

    def forward(self, input, mask):
        # input: (bsz, seq_len, hdim)  mask: (bsz, seq_len)

        lengths = (mask.sum(dim=1)).data.cpu().numpy().astype('int')
        output = utils.run_rnn(input, self.bilstm, lengths.squeeze(-1))

        # output: (bsz, seq_len, hdim)
        output_forward = output[:, :, :self.hidden_dim]
        output_backward = output[:, :, self.hidden_dim:]

        # vec_sema: (bsz, seq_len, sema_dim)
        # vec_stru: (bsz, seq_len, stru_dim)
        vec_sema = torch.cat([output_forward[:, :, :self.sema_dim // 2],
                              output_backward[:, :, :self.sema_dim // 2]], dim=-1)
        vec_stru = torch.cat([output_forward[:, :, self.sema_dim // 2:],
                              output_backward[:, :, self.sema_dim // 2:]], dim=-1)

        # tp/tc: (bsz, seq_len, stru_dim)
        tp = torch.tanh(self.d2tp(vec_stru))
        tc = torch.tanh(self.d2tc(vec_stru))

        # mask: (seq_len, bsz, 1)
        mask_sq = mask.matmul(mask.transpose(1, 2))

        # f: (bsz, seq_len, seq_len)
        # f_r: (bsz, seq_len, 1)
        f = tp.matmul(self.w_a). \
            matmul(tc.transpose(1, 2))
        f_r = self.w_root(vec_stru)
        f = f.exp() * mask_sq
        f_r = f_r.exp() * mask
        seq_len = f.shape[1]
        f[:, range(seq_len), range(seq_len)] = 0

        # a0: (bsz, 1, seq_len)
        # a: (bsz, seq_len, seq_len)
        a0, a = self.struct_atten(f, f_r)
        a_c = a

        # a_p: (bsz, seq_len, seq_len + 1)
        a_p = torch.cat([a0, a], dim=1).transpose(1, 2)

        root_embs = self.root_emb.expand_as(vec_sema[:, :1, :])
        # vec_sem_root: (bsz, seq_len + 1, sema_dim)
        vec_sem_root = torch.cat([root_embs, vec_sema], dim=1)

        # c: (bsz, seq_len, sema_dim)
        # p: (bsz, seq_len, sema_dim)
        c = torch.matmul(a_c, vec_sema)
        p = torch.matmul(a_p, vec_sem_root)

        # output: (bsz, seq_len, sema_dim)
        output = torch.tanh(self.w_r(torch.cat([vec_sema, p, c], dim=-1)))
        output = output * mask
        emb = torch.max(output, 1)[0]

        return {'h': emb,
                'nodes:': output}