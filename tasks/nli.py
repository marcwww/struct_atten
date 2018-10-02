import torch
from torch import nn
from torch.nn import functional as F

class InterAttention(nn.Module):

    def __init__(self, sema_dim, dropout):

        super(InterAttention, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(sema_dim, sema_dim),
                                 nn.LeakyReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(sema_dim, sema_dim),
                                 nn.LeakyReLU())
        self.compare = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(sema_dim * 2, sema_dim),
                                     nn.LeakyReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(sema_dim, sema_dim),
                                     nn.LeakyReLU())
        self.inf = -1e10

    def forward(self, r1, r2, mask1, mask2):
        # mask: (bsz, seq_len)
        # r: (bsz, seq_len, sema_dim)
        r1 = self.mlp(r1)
        r2 = self.mlp(r2)

        mask1_ex = mask1.float().expand_as(r1)
        mask2_ex = mask2.float().expand_as(r2)

        r1 = r1 * mask1_ex
        r2 = r2 * mask2_ex

        # lens: (bsz,)
        lens1 = mask1.sum(dim=1)
        lens2 = mask2.sum(dim=1)

        mask = torch.matmul(mask1_ex, mask2_ex.transpose(1, 2))
        mask_inf = mask.clone()
        mask_inf.masked_fill_(mask_inf.eq(0), self.inf)
        mask_inf.masked_fill_(mask_inf.gt(0), 0)

        # o: (bsz, seq_len1, seq_len2)
        o = torch.matmul(r1, r2.transpose(1, 2))
        o = o + mask_inf

        mask_matrix = mask.gt(0)
        o1 = F.softmax(o, dim=2) * mask_matrix.float()
        o2 = F.softmax(o, dim=1) * mask_matrix.float()

        # r_c: (bsz, seq_len, sema_dim)
        r1_c = torch.matmul(o1, r2)
        r2_c = torch.matmul(o2.transpose(1, 2), r1)

        # r_compare: (bsz, seq_len, sema_dim)
        r1_compare = self.compare(torch.cat([r1, r1_c], dim=-1))
        r2_compare = self.compare(torch.cat([r2, r2_c], dim=-1))

        # new mask for concatted vectors
        mask1_ex = mask1.float().expand_as(r1_compare)
        mask2_ex = mask2.float().expand_as(r2_compare)

        # r_pooling: (bsz, sema_dim)
        # r1_pooling = (r1_compare * mask1_ex).sum(1)
        # r2_pooling = (r2_compare * mask2_ex).sum(1)
        r1_pooling = (r1_compare * mask1_ex).sum(1) / lens1.float()
        r2_pooling = (r2_compare * mask2_ex).sum(1) / lens2.float()

        return r1_pooling, r2_pooling

class NLI(nn.Module):

    def __init__(self, encoder, embedding, dropout):

        super(NLI, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.padding_idx = encoder.padding_idx
        sema_dim = encoder.sema_dim
        self.mlp = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(sema_dim * 2, sema_dim),
                                 nn.LeakyReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(sema_dim, 3),
                                 nn.LeakyReLU())
        self.inter_atten = InterAttention(sema_dim, dropout)
        self.emb_affine = nn.Sequential(nn.Linear(embedding.embedding_dim, encoder.edim),
                                        nn.Dropout(dropout))

    def forward(self, seq1, seq2):
        seq1 = seq1.transpose(0, 1)
        seq2 = seq2.transpose(0, 1)

        embs1 = self.embedding(seq1)
        embs2 = self.embedding(seq2)

        embs1 = self.emb_affine(embs1)
        embs2 = self.emb_affine(embs2)

        # mask: (bsz, seq_len, 1)
        mask1 = seq1.data.ne(self.padding_idx).unsqueeze(-1)
        mask2 = seq2.data.ne(self.padding_idx).unsqueeze(-1)

        # r: (bsz, seq_len, sema_dim)
        out1 = self.encoder(embs1, mask1.float())
        out2 = self.encoder(embs2, mask2.float())
        r1 = out1['nodes']
        r2 = out2['nodes']
        if 'mask' in out1 and 'mask' in out2:
            mask1 = out1['mask'].unsqueeze(-1)
            mask2 = out2['mask'].unsqueeze(-1)

        # mask for inter_atten...
        # r_pooling: (bsz, sema_dim * 2)
        r1_pooling, r2_pooling = self.inter_atten(r1, r2, mask1, mask2)
        r = torch.cat([r1_pooling, r2_pooling], dim=-1)

        # output: (bsz, 3)
        output = self.mlp(r)
        return output