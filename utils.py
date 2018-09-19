import numpy as np
from torch.nn.init import xavier_uniform_, normal_
import torch
from torch import nn
from torch.nn import functional as F
import logging
import random
import time
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import crash_on_ipy
from block import block_diag

LOGGER = logging.getLogger(__name__)

def run_rnn(input, rnn, lengths):
    sorted_idx = np.argsort(lengths)[::-1].tolist()
    rnn_input = pack_padded_sequence(input[sorted_idx], lengths[sorted_idx], batch_first=True)
    rnn_out, _ = rnn(rnn_input)  # (bsize, ntoken, hidsize*2)
    rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
    rnn_out = rnn_out[np.argsort(sorted_idx).tolist()]

    return rnn_out

def LU(A, eps = 1e-10):
    assert len(A.shape) == 3 and \
           A.shape[1] == A.shape[2]
    n = A.shape[1]
    U = A.clone()
    L = A.new_zeros(A.shape)
    L[:, range(n), range(n)] = 1

    for k in range(n-1):
        for j in range(k+1, n):
            L[:, j, k] = U[:, j, k] / (U[:, k, k] + eps)
            U[:, j, k:] -= L[:, j, k].unsqueeze(-1) * U[:, k, k:]

    return L, U

def inv3(A):
    bsz, n = A.shape[0], A.shape[1]
    A_inv = A.clone()
    A_diag_inv = A.new_zeros((bsz*n, bsz*n))
    for b in range(bsz):
        start = n*b
        end = n*b+n
        A_diag_inv[start:end, start:end] = A_inv[b]

    A_diag_inv = torch.inverse(A_diag_inv)

    for b in range(bsz):
        start = n*b
        end = n*b+n
        A_inv[b] = A_diag_inv[start:end, start:end]

    return A_inv

def inv2(A, eps=1e-4):
    bsz, n = A.shape[0], A.shape[1]
    indices_bsz, indices_n, indices_diag = \
        range(bsz), range(n), [[i for _ in range(bsz)] for i in range(n)]
    A_copy = A.clone().data
    A_inv = A.new_zeros(A.shape).data
    A_inv[:, indices_n, indices_n] = 1
    # min_val = A.new_ones((1,)) * 1e-5

    for i in range(n):
        indices_max = torch.max(A_copy[:, i:, i], dim=1)[1] + i

        # swap the max to the diagonal
        A_copy[indices_bsz, indices_diag[i]], A_copy[indices_bsz, indices_max] = \
            A_copy[indices_bsz, indices_max], A_copy[indices_bsz, indices_diag[i]]
        A_inv[indices_bsz, indices_diag[i]], A_inv[indices_bsz, indices_max] =\
            A_inv[indices_bsz, indices_max], A_inv[indices_bsz, indices_diag[i]]

        # begin gaussian
        indices = list(range(n))
        indices.pop(i)
        divisor = A_copy[:, i:i+1, i:i+1].\
            masked_fill_(A_copy[:, i:i+1, i:i+1].eq(0), eps)
        factor = A_copy[:, indices, i:i+1] / divisor

        A_copy[:, indices, :] -= factor.matmul(A_copy[:, i:i + 1, :])
        A_inv[:, indices, :] -= factor.matmul(A_inv[:, i:i+1, :])

    divisor = A_copy[:, indices_n, indices_n].unsqueeze(-1)
    A_inv /= divisor
    A_copy /= divisor

    A_inv_grad = - A_inv.matmul(A).matmul(A_inv)
    return A_inv + A_inv_grad - A_inv_grad.data

def inv(A, eps = 1e-4):

    assert len(A.shape) == 3 and \
           A.shape[1] == A.shape[2]
    n = A.shape[1]
    U = A.clone().data
    zero_mask = U.eq(0)
    U.masked_fill_(zero_mask, eps)
    L = A.new_zeros(A.shape).data
    L[:, range(n), range(n)] = 1
    I = L.clone()

    # A = LU
    # [A I] = [LU I] -> [U L^{-1}]
    L_inv = I
    for i in range(n-1):
        L[:, i+1:, i:i+1] = U[:, i+1:, i:i+1] / U[:, i:i+1, i:i+1]
        L_inv[:, i+1:, :] -= L[:, i+1:, i:i+1].matmul(L_inv[:, i:i+1, :])
        U[:, i+1:, :] -= L[:, i+1:, i:i+1].matmul(U[:, i:i+1, :])
        print(i, U[:, i+1:, :])

    # [U L^{-1}] -> [I U^{-1}L^{-1}] = [I (LU)^{-1}]
    A_inv = L_inv
    for i in range(n-1, -1, -1):
        A_inv[:, i:i+1, :] = A_inv[:, i:i+1, :] / U[:, i:i+1, i:i+1]
        U[:, i:i+1, :] = U[:, i:i+1, :] / U[:, i:i+1, i:i+1]

        if i > 0:
            A_inv[:, :i, :] -= U[:, :i, i:i+1].matmul(A_inv[:, i:i+1, :])
            U[:, :i, :] -= U[:, :i, i:i+1].matmul(U[:, i:i+1, :])

    A_inv_grad = - A_inv.matmul(A).matmul(A_inv)
    return A_inv + A_inv_grad - A_inv_grad.data

def modulo_convolve(w, s):
    # w: (bsz, N)
    # s: (bsz, 3)
    bsz, ksz = s.shape
    assert ksz == 3

    # t: (1, bsz, 1+N+1)
    t = torch.cat([w[:,-1:], w, w[:,:1]], dim=-1).\
        unsqueeze(0)
    device = s.device
    kernel = torch.zeros(bsz, bsz, ksz).to(device)
    kernel[range(bsz), range(bsz), :] += s
    # c: (bsz, N)
    c = F.conv1d(t, kernel).squeeze(0)
    return c

def split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

def one_hot_matrix(stoi, device, edim):

    assert len(stoi) <= edim, \
        'embedding dimension must be larger than voc_size'

    voc_size = len(stoi)
    res = torch.zeros(voc_size,
                      edim,
                      requires_grad=False)
    for i in range(voc_size):
        res[i][i] = 1

    return res.to(device)

def shift_matrix(n):
    W_up = np.eye(n)
    for i in range(n-1):
        W_up[i,:] = W_up[i+1,:]
    W_up[n-1,:] *= 0
    W_down = np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:] = W_down[i-1,:]
    W_down[0,:] *= 0
    return W_up,W_down

def avg_vector(i, n):
    V = np.zeros(n)
    V[:i+1] = 1/(i+1)
    return V

def init_model_xavier(model):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

def init_model_normal(model):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            normal_(p, mean=0, std=0.01)

def init_embedding_normal(emb):
    normal_(emb.weight, mean=0, std=1)

def progress_bar(percent, loss, epoch):
    """Prints the progress until the next report."""

    fill = int(percent * 40)
    str_disp = "\r[%s%s]: %.2f/epoch %d" % ('=' * fill,
                                         ' ' * (40 - fill),
                                         percent,
                                         epoch)
    for k, v in loss.items():
        str_disp += ' (%s:%.4f)' % (k, v)

    print(str_disp, end='')

def seq_lens(seq, padding_idx):
    mask = seq.data.eq(padding_idx)
    len_total, bsz = seq.shape
    lens = len_total - mask.sum(dim=0)
    return lens

class Attention(nn.Module):
    def __init__(self, hdim):
        super(Attention, self).__init__()
        self.hc2ha = nn.Sequential(nn.Linear(hdim * 2, hdim, bias=False),
                                  nn.Tanh())

    def forward(self, h, enc_outputs):
        # h: (1, bsz, hdim)
        # h_current: (1, bsz, 1, hdim)
        h_current = h.unsqueeze(2)
        # enc_outputs: (len_total, bsz, hdim, 1)
        enc_outputs = enc_outputs.unsqueeze(-1)
        # a: (len_total, bsz, 1, 1)
        a = h_current.matmul(enc_outputs)
        a = F.softmax(a, dim=0)
        # c: (len_total, bsz, hdim, 1)
        c = a * enc_outputs
        # c: (bsz, hdim)
        c = c.sum(0).squeeze(-1).unsqueeze(0)
        ha = self.hc2ha(torch.cat([h, c], dim=-1))
        ha = F.tanh(ha)
        return ha

def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

def gumbel_softmax_sample(logits, tau, hard, eps=1e-10):

    shape = logits.size()
    assert len(shape) == 2
    y_soft = F._gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        bsz, N = y_soft.shape
        k = []
        for b in range(bsz):
            idx = np.random.choice(N, p=y_soft[b].data.cpu().numpy())
            k.append(idx)
        k = np.array(k).reshape(-1, 1)
        k = y_soft.new_tensor(k, dtype=torch.int64)

        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

def gumbel_sigmoid_sample(logit, tau, hard):

    shape = logit.shape
    assert (len(shape) == 2 and shape[-1] == 1) \
            or len(shape) == 1

    if len(shape) == 1:
        logit = logit.unsqueeze(-1)
        shape = logit.shape

    zero = logit.new_zeros(*shape)
    res = torch.cat([logit, zero], dim=-1)
    res = gumbel_softmax_sample(res, tau=tau, hard=hard)

    return res[:, 0]

def gumbel_sigmoid_max(logit, tau, hard):

    shape = logit.shape
    assert (len(shape) == 2 and shape[-1] == 1) \
            or len(shape) == 1

    if len(shape) == 1:
        logit = logit.unsqueeze(-1)
        shape = logit.shape

    zero = logit.new_zeros(*shape)
    res = torch.cat([logit, zero], dim=-1)
    res = F.gumbel_softmax(res, tau=tau, hard=hard)

    return res[:, 0]

def param_str(opt):
    res_str = {}
    for attr in dir(opt):
        if attr[0] != '_':
            res_str[attr] = getattr(opt, attr)
    return res_str

def time_int():
    return int(time.time())

def load_pretrain(embed, vectors):
    # init_embedding_normal(embed)

    mask = vectors.ne(0).sum(-1).ne(0)
    idices = []
    for idx, mask_val in enumerate(mask):
        if mask_val != 0:
            idices.append(idx)
    embed.weight.data[idices] = \
        vectors[idices].to(embed.weight.device)

if __name__ == '__main__':
    # up, down = shift_matrix(3)
    # x = np.array([[0,1,2]]).transpose()
    # print(x)
    # print(up.dot(x))
    # print(down)
    # A = [[1, -2, -2, -3],
    #      [3, -9, 0, -9],
    #      [-1, 2, 4, 7],
    #      [-3, -6, 26, 2]]
    # A = torch.randn(2, 4, 4)
    # LU_deco_inverse(np.mat(A))
    # A = torch.Tensor(A).unsqueeze(0)
    # inv(A)
    # print(inv(A)[0])
    # print(torch.inverse(A[0]))

    # for _ in range(100):
    #     A = torch.randn(1, 3, 3)
    #     print(inv2(A).matmul(A).sum())
        # print(torch.inverse(A[0]))
        # cost = torch.norm(inv2(A).unsqueeze(0) - torch.inverse(A[0])).sum()
        # print(cost)

    # import pickle
    # A = pickle.load(open('LL.pkl', 'rb'))
    # A = torch.Tensor(A)
    # print(np.linalg.matrix_rank(A[30]+torch.randn(22, 22)*1000))
    # print(inv2(A[30].unsqueeze(0)))
    # print(inv2(A[30, None]).matmul(A[30, None]))
    # print(torch.inverse(A[30]).matmul(A[30]).sum())
    # A = torch.\
    #     tensor([[[ 1.,  2.,  9., 3],
    #      [ 4.,  9.,  4., 5],
    #      [ 1.,  1.,  1., 1],
    #      [1., 2., 1., 1]]])
    A = torch.\
        tensor([[[ 1.,  2.,  9.],
         [ 2.,  9.,  4.],
         [ 3.,  9.,  6.]]])
    B = torch. \
        tensor([[[3., 2., 9.],
                 [1., 2., 7.],
                 [3., 5., 8.]]])

    # A += 1e-4
    print(np.linalg.matrix_rank(B))
    print(inv2(torch.cat([A,B])))
    # print(inv2(A)[1])
    # print(inv2(A).matmul(B))
    print(torch.inverse(A[0]))
    print(torch.inverse(B[0]))

    import time
    begin = time.time()
    inv3(A)
    a = time.time()
    print(a - begin)

    inv2(A)
    # torch.inverse(A[0])
    b = time.time()
    print(b - a)

    # print(torch.inverse(A[0]).matmul(A[0]))

    # np.set_printoptions(precision=4)
    #
    # A = torch.\
    #     tensor([[[ 0.,  2.,  9.],
    #      [ 4.,  9.,  4.],
    #      [ 3.,  9.,  6.]]])
    #
    # A_inv, L, U = inv(A)
    # print(A_inv)
    # print('L:')
    # print(L)
    # print('U:')
    # print(U)
    #
    # print('---'*20)
    # L, U = LU(A)
    # print('L:')
    # print(L)
    # print('U:')
    # print(U)
    #
    # print('---' * 20)
    # A_inv = torch.inverse(A[0])
    # print(A_inv)
    #
    # print('---' * 20)
    # LU_deco_inverse(np.mat(A.squeeze(0).data.numpy()))