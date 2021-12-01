# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from multiprocessing import Pool
import os

import torch

from fairseq.tokenizer import tokenize_line
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils, Dictionary


class BertDictionaryNAR(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        bos='<s>',
        extra_special_symbols=None,
    ):
        super().__init__(pad, eos, unk, bos, extra_special_symbols)

    @classmethod
    def load_from_file(cls, filename):
        d = cls()
        d.symbols = []
        d.count = []
        d.indices = {}

        with open(filename, 'r', encoding='utf-8', errors='ignore') as input_file:
            for line in input_file:
                k, v = line.split(' ')
                d.add_symbol(k)

        d.unk_word = '[UNK]'
        d.pad_word = '[PAD]'
        d.eos_word = '[SEP]'
        d.bos_word = '[CLS]'

        d.bos_index = d.add_symbol('[CLS]')
        d.pad_index = d.add_symbol('[PAD]')
        d.eos_index = d.add_symbol('[SEP]')
        d.unk_index = d.add_symbol('[UNK]')

        d.nspecial = 999
        return d

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols, ex_vals + self.count))

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, 'bos_index'):
            sent = ' '.join(token_string(i) for i in tensor)
        else:
            sent = ' '.join(token_string(i) for i in tensor)
        return data_utils.process_bpe_symbol(sent, bpe_symbol)