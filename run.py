#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import pickle
import os
import logging
import torch
from dataset import AtisDataSet, AtisLoader
from torch.utils.data import DataLoader
from vocab import Vocab
from torch.autograd import Variable
from model import biLSTM
from utils import accuracy, remove_pad, conlleval
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser('Project of COSEM')
    parser.add_argument('--prepare', action='store_true', default=False,
                        help='prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='visualize')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--embed_learning_rate', type=float, default=0.001,
                                help='embedding learning rate')
    train_settings.add_argument('--lr_decay', type=float, default=0.75,
                                help='weight decay')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_rate', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--epoch_num', type=int, default=20,
                                help='train epoch_num')
    train_settings.add_argument('--begin_epoch', type=int, default=1)
    train_settings.add_argument('--batch_step', type=int, default=15)


    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--has_cuda', type=bool, default=False)
    model_settings.add_argument('--embedding_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--num_layers', type=int, default=1,
                                help='num of LSTM layers')
    model_settings.add_argument('--max_len', type=int, default=50,
                                help='max length of sentence')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--slot_name_file', type=str,
                               default='./data/atis_slot_names.xlsx')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/atis.train.txt'])
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/atis.test.txt'])
    path_settings.add_argument('--vocab_dir', default='./data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--save_path', default='./savedmodel/',
                               help='path to save model')
    return parser.parse_args()

def prepare(args):
    """
        checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("hit-cosem-2018")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary ...')
    atis_data = AtisDataSet(args.max_len, args.slot_name_file, train_files=args.train_files, test_files=args.test_files)
    vocab = Vocab()
    for word in atis_data.word_iter():
        vocab.add(word)
    logger.info("Unfiltered vocab size is {}".format(vocab.size()))
    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')

def train(args):
    logger = logging.getLogger("hit-cosem-2018")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    atis_data = AtisDataSet(args.max_len, args.slot_name_file, train_files=args.train_files, test_files=args.test_files)
    logger.info('Converting text into ids...')
    atis_data.convert_to_ids(vocab)
    atis_data.dynamic_padding(vocab.token2id[vocab.pad_token])
    train_data, test_data = atis_data.get_numpy_data()
    train_data = AtisLoader(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_data = AtisLoader(test_data)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    model = biLSTM(args, vocab.size(), len(atis_data.slot_names))
    optimizer = model.get_optimizer(args.learning_rate, args.embed_learning_rate, args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    score = []
    losses = []
    for eidx, _ in enumerate(range(args.epoch_num), 1):
        for bidx, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            sentences, labels = data
            if args.has_cuda:
                sentences, labels = Variable(sentences).cuda(), Variable(labels).cuda()
            else:
                sentences, labels = Variable(sentences), Variable(labels)
            output = model(sentences)
            labels = labels.view(-1)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            predicted = predicted.cpu().numpy()
            labels = labels.data.cpu().numpy()
            losses.append(loss.data[0])
            logger.info('epoch: {} batch: {} loss: {:.4f} acc: {:.4f}'.format(eidx, bidx, loss.data[0],
                                                                          accuracy(predicted, labels)))
        if eidx >= args.begin_epoch:
            if args.embed_learning_rate == 0:
                args.embed_learning_rate = 2e-4
            elif args.embed_learning_rate > 0:
                args.embed_learning_rate *= args.lr_decay
                if args.embed_learning_rate <= 1e-5:
                    args.embed_learning_rate = 1e-5
            args.learning_rate = args.learning_rate * args.lr_decay
            optimizer = model.get_optimizer(args.learning_rate,
                                            args.embed_learning_rate,
                                            args.weight_decay)

        logger.info('do eval on test set...')
        f = open("./result/result_epoch"+str(eidx)+".txt", 'w', encoding='utf-8')
        for data in test_loader:
            sentences, labels = data
            if args.has_cuda:
                sentences, labels = Variable(sentences).cuda(), Variable(labels).cuda()
            else:
                sentences, labels = Variable(sentences), Variable(labels)  # batch_size * max_len
            output = model(sentences)

            sentences = sentences.data.cpu().numpy().tolist()
            sentences = [vocab.recover_from_ids(remove_pad(s, vocab.token2id[vocab.pad_token]))
                         for s in sentences]
            labels = labels.data.cpu().numpy().tolist()
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.view(-1, args.max_len).cpu().numpy().tolist()
            iter = [zip(s,
                        map(lambda x:atis_data.slot_names[x], labels[i][:len(s)]),
                               map(lambda x:atis_data.slot_names[x], predicted[i][:len(s)])
                        ) for i, s in enumerate(sentences)]
            for it in iter:
                for z in it:
                    z = list(map(str, z))
                    f.write(' '.join(z)+'\n')
        f.close()
        score.append(conlleval(eidx))
        torch.save(model.state_dict(), args.save_path+"biLSTM_epoch"+str(eidx)+".model")

    max_score_eidx = score.index(max(score))+1
    logger.info('epoch {} gets max score.'.format(max_score_eidx))
    os.system('perl ./eval/conlleval.pl < ./result/result_epoch' + str(max_score_eidx) + '.txt')
    
    x = [i + 1 for i in range(len(losses))]
    plt.plot(x, losses, 'r')
    plt.xlabel("time_step")
    plt.ylabel("loss")
    plt.title("CrossEntropyLoss")
    plt.show()

def visualize(args):
    score = []
    x = [i+1 for i in range(args.epoch_num)]
    for eidx in x:
        score.append(conlleval(eidx))
    plt.plot(x, score, 'b',)
    plt.xlabel("epoch_num")
    plt.ylabel("accuracy(%)")
    plt.title("Accuracy")
    plt.show()




def run():
    args = parse_args()

    logger = logging.getLogger("hit-cosem-2018")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("Running with args : {}".format(args))

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.visualize:
        visualize(args)


if __name__ == "__main__":
    run()
