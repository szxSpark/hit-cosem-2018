#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from tqdm import tqdm
from collections import Counter
import logging

class Vocab(object):

    def __init__(self):
        self.token2id, self.id2token, self.token_cnt = {}, {}, {}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.initial_tokens = [self.pad_token, self.unk_token]

        for token in self.initial_tokens:
            self.add(token)

    def add(self, token, cnt=1):
        """
        adds the token to vocab
        :param token: a string
        :param cnt:  a num indicating the count of the token to add, default is 1
        :return: idx
        """
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def size(self):
        return len(self.id2token)

    def get_id(self, token):
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def convert_to_ids(self, tokens):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        :param tokens:  tokens: a list of token
        :return: a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        :param ids: a list of ids to convert
        :param stop_id: the stop id, default is None
        :return: a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens

    def filter_tokens_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)