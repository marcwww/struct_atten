import torchtext
import pickle
import os
from macros import *
from torchtext.data import Dataset

class Example(object):

    def __init__(self, seq1, seq2, lbl):
        self.seq1 = self.tokenizer(seq1)
        self.seq2 = self.tokenizer(seq2)
        self.lbl = lbl

    def tokenizer(self, seq):
        return list(seq)

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            gold_label, \
            _, _, _, _, \
            sentence1, sentence2,\
            _, _, _, _, _, _, _ = line.split('\t')

            if gold_label == NOT_SURE:
                continue

            examples.append(Example(sentence1, sentence2, LBL[gold_label]))

    return examples

def build_iters(ftrain, fvalid, bsz, device):

    examples_train = load_examples(ftrain)
    print('Done loading.')

    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=None)
    LBL = torchtext.data.Field(sequential=False, use_vocab=False)

    train = Dataset(examples_train, fields=[('seq1', SEQ),
                                            ('seq2', SEQ),
                                            ('lbl', LBL)])
    SEQ.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('seq1', SEQ),
                                            ('seq2', SEQ),
                                            ('lbl', LBL)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq1),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq1),
                                         sort_within_batch=True,
                                         device=device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'SEQ': SEQ,
            'LBL': LBL}


if __name__ == '__main__':
    ftrain = os.path.join(DATA, 'snli_1.0_train.txt')
    fvalid = os.path.join(DATA, 'snli_1.0_dev.txt')

    iters = build_iters(ftrain, fvalid, 32, -1)