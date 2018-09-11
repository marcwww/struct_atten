import numpy as np
from torch.nn.init import xavier_uniform_
import torch
from torch import nn
from torch.nn import functional as F
import logging
import random
import time
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import crash_on_ipy

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

def inv(A, eps = 1e-4):

    assert len(A.shape) == 3 and \
           A.shape[1] == A.shape[2]
    n = A.shape[1]
    U = A.clone().data + eps
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

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

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


def LU_deco_inverse(m):
    dim = m.shape[0]
    E = np.mat(np.eye(dim))
    L = np.mat(np.eye(dim))
    U = m.copy()
    U += 1e-3
    for i in range(dim):
        # 上面L=m.copy()时用这个，然后我们将其改进先使其初始为单位阵
        # L[:i,i] = 0
        # L[i:dim,i] = U[i:dim,i] / U[i,i]
        L[i + 1:, i] = U[i + 1:, i] / U[i, i]

        # E[i+1:dim,i+1:dim] = E[i+1:dim,i+1:dim] - L[i+1:dim,i]*E[i,i+1:dim]
        # 行变换应该是整个一行的变，而不是上面写的变部分，另E的变换一定要在U之前。
        # 这里还将dim去掉因为意思就是从j+1到最后一个元素，可省略dim看起来没那么晕。
        E[i + 1:, :] = E[i + 1:, :] - L[i + 1:, i] * E[i, :]
        U[i + 1:, :] = U[i + 1:, :] - L[i + 1:, i] * U[i, :]
    # U[i+1:dim,:i+1] = 0
    # U[i+1:dim,i+1:dim] = U[i+1:dim,i+1:dim] - L[i+1:dim,i]*U[i,i+1:dim]
    # 上面这个这样写不划算采用上上面那句代替这俩

    print("\nLU分解后的L,U矩阵:")
    print("L=", L)
    print("U=", U)
    print("将A化为上三角阵U后随之变换的E矩阵:")
    print("E=", E)

    # 普通从最后一行开始消去该列的for循环
    # U = U.copy()
    # for i in range(dim-1,-1,-1):
    # 	E[i,:] = E[i,:]/U[i,i]
    # 	U[i,:] = U[i,:]/U[i,i]
    # 	for j in range(i-1,-1,-1):
    # 		E[j,:] = E[j,:] - U[j,i]*E[i,:]
    # 		U[j,:] = U[j,:] - U[j,i]*U[i,:]

    # 写成向量形式
    # U = U.copy()
    # for i in range(dim-1,-1,-1):
    # 	E[i,:] = E[i,:]/U[i,i]
    # 	U[i,:] = U[i,:]/U[i,i]

    # 	E[i-1:-1:-1,:] = E[i-1:-1:-1,:] - U[i-1:-1:-1,i]*E[i,:]
    # 	U[i-1:-1:-1,:] = U[i-1:-1:-1,:] - U[i-1:-1:-1,i]*U[i,:]

    # 通过观察做行变换的过程中发现的规律，比上面注释掉的方法更简单
    E1 = np.mat(np.eye(dim))  # 这个E1用来求U的逆
    for i in range(dim - 1, -1, -1):
        # 对角元单位化
        E[i, :] = E[i, :] / U[i, i]
        E1[i, :] = E1[i, :] / U[i, i]
        U[i, :] = U[i, :] / U[i, i]

        E[:i, :] = E[:i, :] - U[:i, i] * E[i, :]
        E1[:i, :] = E1[:i, :] - U[:i, i] * E1[i, :]
        U[:i, :] = U[:i, :] - U[:i, i] * U[i, :]  # r_j = m_ji - r_j*r_i

    print("\n将上三角阵U变为单位阵后的U和随之变换后的E分别为:")
    print("U=", U)
    print("E=", E)
    print("使用系统自带的求inverse方法得到的逆为:")
    print("m_inv=", m.I)
    print("\nU的逆E1为:")
    print("E1=", E1)

    # 当然，我们还可以来求一下下三角阵L的逆
    E2 = np.mat(np.eye(dim))
    for i in range(dim):
        # 因为这里对角元已经是1了就不做对角元单位化这部了
        E2[i + 1:, :] = E2[i + 1:, :] - L[i + 1:, i] * E2[i, :]
        L[i + 1:, :] = L[i + 1:, :] - L[i + 1:, i] * U[i, :]

    print("\n将下三角阵L变为单位阵后的L和随之变换后的E2分别为:")
    print("L=", L)
    print("E2=", E2)

    print("\n由A=LU,得A逆=U的逆*L的逆")
    print("U的逆E1*L的逆E2=", E1 * E2)

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

    for _ in range(100):
        A = torch.randn(1, 3, 3)
        print(inv(A).unsqueeze(0))
        print(torch.inverse(A[0]))
        cost = torch.norm(inv(A).unsqueeze(0) - torch.inverse(A[0])).sum()
        print(cost)

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