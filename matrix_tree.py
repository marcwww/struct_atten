import torch
import utils
import tensorflow as tf
import numpy as np

def mt_onmt(input, r, eps = 1e-5):

    laplacian = input + eps
    output = input.clone()
    for b in range(input.size(0)):
        lap = laplacian[b].masked_fill(
            torch.eye(input.size(1)).ne(0), 0)
        lap = -lap + torch.diag(lap.sum(0))
        # store roots on diagonal
        lap[0] = r
        inv_laplacian = lap.inverse()

        factor = inv_laplacian.diag().unsqueeze(1) \
            .expand_as(input[b]).transpose(0, 1)
        term1 = input[b].mul(factor).clone()
        term2 = input[b].mul(inv_laplacian.transpose(0, 1)).clone()
        term1[:, 0] = 0
        term2[0] = 0
        output[b] = term1 - term2
        roots_output = r[b].mul(
            inv_laplacian.transpose(0, 1)[0])
        output[b] = output[b] + torch.diag(roots_output.squeeze(0))

    seq_len = output.shape[1]
    roots = output[:, range(seq_len), range(seq_len)].clone()
    output[:, range(seq_len), range(seq_len)] = 0
    output = torch.cat([roots.unsqueeze(1), output], dim=1)

    return output

def mt_mine(A, r):
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

    return d, LL, LL_inv, term1, term2

def mt_ly(r, A, mask1, mask2):
    L = tf.reduce_sum(A, 1)
    L = tf.matrix_diag(L)
    L = L - A
    LL = L[:, 1:, :]
    LL = tf.concat([tf.expand_dims(r, [1]), LL], 1)
    LL_inv = tf.matrix_inverse(LL)  # batch_l, doc_l, doc_l
    d0 = tf.multiply(r, LL_inv[:, :, 0])
    LL_inv_diag = tf.expand_dims(tf.matrix_diag_part(LL_inv), 2)
    tmp1 = tf.matrix_transpose(tf.multiply(tf.matrix_transpose(A), LL_inv_diag))
    tmp2 = tf.multiply(A, tf.matrix_transpose(LL_inv))
    d = tmp1 * mask1 - tmp2 * mask2
    d = tf.concat([tf.expand_dims(d0, [1]), d], 1)
    return d, LL, LL_inv, tmp1 * mask1, tmp2 * mask2


if __name__ == '__main__':

    A = tf.constant([[[0., 4., 4., 5., 7.],
         [3., 0., 6., 3., 2.],
         [4., 4., 0., 0., 5.],
         [7., 8., 1., 0., 4.],
         [1., 2., 1., 8., 0.]]])

    r = tf.constant([[4., 8., 6., 1., 6.]])

    mask_parser_1 = np.ones([1, 5, 5], np.float32)
    mask_parser_2 = np.ones([1, 5, 5], np.float32)
    mask_parser_1[:, :, 0] = 0
    mask_parser_2[:, 0, :] = 0

    res = mt_ly(r, A, mask_parser_1, mask_parser_2)

    sess = tf.Session()
    d, LL, LL_inv, tmp1, tmp2 = sess.run(res)

    # np.set_printoptions(precision=4)
    print('--'*20+'mt_ly'+'--'*20)
    print('d0')
    print(torch.Tensor(d)[:, 0, :])
    print('d')
    print(torch.Tensor(d)[:, 1:, :])
    print('LL')
    print(torch.Tensor(LL))
    print('LL_inv')
    print(torch.Tensor(LL_inv))
    print('tmp1')
    print(torch.Tensor(tmp1))
    print('tmp2')
    print(torch.Tensor(tmp2))

    print('--' * 20 + 'mt_mine' + '--' * 20)
    A = torch.tensor([[0., 4., 4., 5., 7.],
         [3., 0., 6., 3., 2.],
         [4., 4., 0., 0., 5.],
         [7., 8., 1., 0., 4.],
         [1., 2., 1., 8., 0.]])

    r = torch.tensor([[[4., 8., 6., 1., 6.]]])

    d, LL, LL_inv, tmp1, tmp2 = mt_mine(A.unsqueeze(0), r.transpose(1, 2))
    print('d0')
    print(torch.Tensor(d)[:, 0, :])
    print('d')
    print(torch.Tensor(d)[:, 1:, :])
    print('LL')
    print(torch.Tensor(LL))
    print('LL_inv')
    print(torch.Tensor(LL_inv))
    print('tmp1')
    print(torch.Tensor(tmp1))
    print('tmp2')
    print(torch.Tensor(tmp2))

    print('--' * 20 + 'mt_onmt' + '--' * 20)
    res = mt_onmt(A.unsqueeze(0), r.squeeze(-1))
    print(res)




