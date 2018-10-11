import torchtext
import pickle
import os
from macros import *
from torchtext.data import Dataset
import torch
# from nltk import word_tokenize
import spacy
from tree_batch import sexpr_to_tree, Forest
from spacy.lang.en import English
import numpy as np
spa_tok = English().Defaults.create_tokenizer()

class Example(object):

    def __init__(self, seq1, seq2, sexp1, sexp2, lbl):
        self.seq1 = self.tokenizer(seq1)
        self.seq2 = self.tokenizer(seq2)
        self.tree1 = sexp1
        self.tree2 = sexp2
        self.lbl = lbl

    def tokenizer(self, seq):
        # return seq.split()
        return [sent.string.strip() for sent in spa_tok(seq)]
        # return word_tokenize(seq)

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            gold_label, \
            _, _, sexp1, sexp2, \
            sentence1, sentence2,\
            _, _, _, _, _, _, _ = line.split('\t')

            if gold_label == NOT_SURE:
                continue

            examples.append(Example(sentence1, sentence2,
                                    sexp1, sexp2,
                                    LBL[gold_label]))

    return examples

class ForestIterator(object):

    def __init__(self, stoi, examples, bsz, device, shuffle=True):
        self.stoi = stoi
        self.seq_num = len(examples)
        self.examples = examples
        self.bsz = bsz
        self.device = device
        self.shuffle = shuffle
        self.batch_idx = 0
        self.batch_num = int(self.seq_num / bsz)
        self.tree_cache = {}

    def __len__(self):
        return self.batch_num

    def __iter__(self):
        return self

    def _restart(self):
        self.batch_idx = 0
        self.seq_indices = np.random.choice(self.seq_num, self.seq_num)

    def __next__(self):
        if self.batch_idx < self.batch_num:
            base = self.batch_idx * self.bsz
            forest1 = []
            forest2 = []
            lbls = []
            for offset in range(self.bsz):
                idx = base + offset
                if idx >= self.seq_num:
                    self._restart()
                    raise StopIteration()

                if idx not in self.tree_cache:
                    tree1 = sexpr_to_tree(self.examples[idx].tree1, self.stoi, self.device)
                    tree2 = sexpr_to_tree(self.examples[idx].tree2, self.stoi, self.device)
                    self.tree_cache[idx] = (tree1, tree2)

                tree1, tree2 = self.tree_cache[idx]
                forest1.append(tree1)
                forest2.append(tree2)
                lbls.append(self.examples[idx].lbl)

            forest1 = Forest(forest1, self.device)
            forest2 = Forest(forest2, self.device)
            lbls = torch.LongTensor(lbls).to(self.device)

            self.batch_idx += 1

            return forest1, forest2, lbls

        self._restart()
        raise StopIteration()

def build_iters(ftrain, fvalid, ftest, bsz, device, pretrain, min_freq):

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
    SEQ.build_vocab(train, vectors=pretrain, min_freq=min_freq)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('seq1', SEQ),
                                            ('seq2', SEQ),
                                            ('lbl', LBL)])

    examples_test = load_examples(ftest)
    test = Dataset(examples_test, fields=[('seq1', SEQ),
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
    test_iter = torchtext.data.Iterator(test, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq1),
                                         sort_within_batch=True,
                                         device=device)

    train_fiter = ForestIterator(SEQ.vocab.stoi, examples_train, bsz, device)
    valid_fiter = ForestIterator(SEQ.vocab.stoi, examples_valid, bsz, device)
    test_fiter = ForestIterator(SEQ.vocab.stoi, examples_test, bsz, device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'test_iter': test_iter,
            'train_fiter': train_fiter,
            'valid_fiter': valid_fiter,
            'test_fiter': test_fiter,
            'SEQ': SEQ,
            'LBL': LBL}

if __name__ == '__main__':
    ftrain = os.path.join(DATA, 'snli_1.0_train.txt')
    fvalid = os.path.join(DATA, 'snli_1.0_dev.txt')

    # iters = build_iters(ftrain, fvalid, 32, -1)

    nlp = spacy.load('en')
    tok = English().Defaults.create_tokenizer(nlp.vocab)