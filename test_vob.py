import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import params
from copy import copy
import torch.backends.cudnn as cudnn
from collections import Counter
import json

train_path = './data/data.txt'
n_vocab = 10000;

class Vocabulary:
    def __init__(self):
        self.itos = list()
        self.stoi = dict()

def load_data(path, vocab):
    with open(path, errors="ignore") as file:
        X = []
        K = []
        y = []
        k = []

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                k = []

            if "your persona:" in line:
                if len(k) == 5:
                    continue
                k_line = line.split("persona:")[1].strip("\n")#.lower()
                k.append(k_line)

            elif "__SILENCE__" not in line:
                K.append(k)
                X_line = " ".join(line.split("\t")[0].split()[1:])#.lower()
                y_line = line.split("\t")[1].strip("\n")#.lower()
                X.append(X_line)
                y.append(y_line)

    X_ind = []
    y_ind = []
    K_ind = []

    for line in X:
        X_temp = []
        #tokens = nltk.word_tokenize(line)
        tokens = line.split()
        for word in tokens:
            if word in vocab.stoi:
                X_temp.append(vocab.stoi[word])
            else:
                X_temp.append(vocab.stoi['<UNK>'])
        X_ind.append(X_temp)

    for line in y:
        y_temp = []
        #tokens = nltk.word_tokenize(line)
        tokens = line.split()
        for word in tokens:
            if word in vocab.stoi:
                y_temp.append(vocab.stoi[word])
            else:
                y_temp.append(vocab.stoi['<UNK>'])
        y_ind.append(y_temp)

    for lines in K:
        K_temp = []
        for line in lines:
            k_temp = []
            #tokens = nltk.word_tokenize(line)
            tokens = line.split()
            for word in tokens:
                if word in vocab.stoi:
                    k_temp.append(vocab.stoi[word])
                else:
                    k_temp.append(vocab.stoi['<UNK>'])
            K_temp.append(k_temp)
        K_ind.append(K_temp)

    return X_ind, y_ind, K_ind

def build_vocab(path, n_vocab):
    with open(path, errors="ignore") as file:
        word_counter = Counter()
        vocab = Vocabulary()
        # vocab = dict()
        # reverse_vocab = dict()
        vocab.stoi['<PAD>'] = params.PAD
        vocab.stoi['<UNK>'] = params.UNK
        vocab.stoi['<SOS>'] = params.SOS
        vocab.stoi['<EOS>'] = params.EOS

        initial_vocab_size = len(vocab.stoi)
        vocab_idx = initial_vocab_size

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                count = 0

            if "your persona:" in line:
                if count == 5:
                    continue
                k_line = line.split("persona:")[1].strip("\n")#.lower()
                #切分单词
                #tokens = nltk.word_tokenize(k_line)

                tokens = k_line.split()
                count += 1

                for word in tokens:
                    if word in vocab.itos:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

            elif "__SILENCE__" not in line:
                X_line = " ".join(line.split("\t")[0].split()[1:])#.lower()
                #tokens = nltk.word_tokenize(X_line)
                tokens = X_line.split()
                for word in tokens:
                    if word in vocab.itos:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

                y_line = line.split("\t")[1].strip("\n")#.lower()
                #tokens = nltk.word_tokenize(y_line)
                tokens = y_line.split()
                for word in tokens:
                    if word in vocab.itos:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

        for key, _ in word_counter.most_common(n_vocab - initial_vocab_size):
            vocab.stoi[key] = vocab_idx
            vocab_idx += 1

        for key, value in vocab.stoi.items():
            vocab.itos.append(key)

    return vocab


vocab = build_vocab(train_path, n_vocab)
# for i in vocab.stoi:
# 	print(i)
# with open('vocab.json', 'w',encoding='utf-8') as fp:
#         json.dump(vocab.stoi, fp)
train_X, train_y, train_K = load_data(train_path, vocab)
print(train_X[0])
print(vocab.itos[9])
print(vocab.itos[10])