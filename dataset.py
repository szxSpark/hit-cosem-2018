#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch.utils import data
import xlrd
from tqdm import tqdm
import logging
from vocab import Vocab
import numpy as np
class AtisDataSet(object):

    def __init__(self, max_len, slot_name_file, train_files=[], test_files=[]):
        self.logger = logging.getLogger("hit-cosem-2018")
        self.max_len = max_len
        self.slot_names = self._load_slot_names(slot_name_file)
        self.logger.info("Slot names : {}.".format(len(self.slot_names)))

        self.train_set, self.test_set = [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file)
            self.logger.info("Train set size: {} sentences.".format(len(self.train_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info("Test set size: {} sentences.".format(len(self.test_set)))


    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        if self.train_set is not None:
            for i in range(len(self.train_set)):
                self.train_set[i][0] = vocab.convert_to_ids(self.train_set[i][0])
                tmp = []
                for l in self.train_set[i][1]:
                    if l != 'O':
                        l = l[2:]
                    tmp.append(self.slot_names.index(l))
                self.train_set[i][1] = tmp

        if self.test_set is not None:
            for i in range(len(self.test_set)):
                self.test_set[i][0] = vocab.convert_to_ids(self.test_set[i][0])
                tmp = []
                for l in self.test_set[i][1]:
                    if l != 'O':
                        l = l[2:]
                    tmp.append(self.slot_names.index(l))
                self.test_set[i][1] = tmp


    def get_numpy_data(self):
        train_sentences, train_labels = None, None
        test_sentences, test_labels = None, None

        if self.train_set is not None:
            train_sentences = np.array([ts[0] for ts in self.train_set], dtype=np.int64)
            train_labels = np.array([ts[1] for ts in self.train_set], dtype=np.int64)

        if self.test_set is not None:
            test_sentences = np.array([ts[0] for ts in self.test_set], dtype=np.int64)
            test_labels = np.array([ts[1] for ts in self.test_set], dtype=np.int64)
        self.logger.info("Successfully build nump data, train_sentences is {0}, test_sentences is {1}".
                         format(np.shape(train_sentences), np.shape(test_sentences)))

        return (train_sentences, train_labels), (test_sentences, test_labels)

    def _load_dataset(self, datapath):
        """
        Loads the dataset
        """
        dataset = []
        with open(datapath, 'r', encoding='utf-8')as fin:
            sentences = []
            labels = []
            for line in tqdm(fin):
                line = line.strip().split()
                assert len(line) == 2 or len(line) == 0
                if len(line) == 2:
                    sentences.append(line[0])
                    labels.append(line[1])
                else:
                    dataset.append([sentences, labels])
                    sentences = []
                    labels = []
        return dataset

    def _load_slot_names(self, datapath):
        """
        Loads slot name from excel
        """

        excelfile = xlrd.open_workbook(datapath)
        sheet = excelfile.sheet_by_index(0)
        slot_names = [sheet.row_values(i)[0] for i in range(sheet.nrows)]
        slot_names.append('O')
        return slot_names

    def dynamic_padding(self, pad_id):
        """
        Dynamically pads with pad_id
        """
        pad_len = self.max_len
        label_O = self.slot_names.index('O')
        if self.train_set is not None:
            self.train_set = [[(ids + [pad_id] * (pad_len - len(ids)))[: pad_len],
                              (labels + [label_O] * (pad_len - len(labels)))[: pad_len]]
                                            for ids, labels in self.train_set]
        if self.test_set is not None:
            self.test_set = [[(ids + [pad_id] * (pad_len - len(ids)))[: pad_len],
                              (labels + [label_O] * (pad_len - len(labels)))[: pad_len]]
                                            for ids, labels in self.test_set]

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample[0]:
                    yield token

class AtisLoader(data.Dataset):

    def __init__(self, data):
        sentences, labels = data
        self.len = sentences.shape[0]
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    a=[1,2,3,4]
    print(a[2:])