import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .data import parse_corpus, format_data
from .model import Net

cropus = 'corpus.txt'
seq_length = 50
batch_size = 32
embedding_dim = 256
hidden_dim = 256
learning_rate = 0.0001
dropout = 0.2
epochs = 2
log_interval = 10
save_interval = 10
output_path = './output/model/ep30/model.bin'
output_c = './output/model/ep30/corpus.bin'

def load_data(path, seq_length, batch_size):
    dataX, dataY, char_to_int, int_to_char, chars = parse_corpus(path, seq_length=seq_length)
    data = format_data(dataX, dataY, n_classes=len(chars), batch_size=batch_size)

    return data, dataX, dataY, char_to_int, int_to_char, chars

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def train(model, optimizer, epoch, data, log_interval):
    model.train()

    for batch_i, (seq_in, target) in enumerate(data):
        seq_in, target = Variable(seq_in), Variable(target)
        optimizer.zero_grad()

        output = model(seq_in)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_i % log_interval == 0:
            print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}'.format(epoch, 100 * batch_i / len(data), loss.data.item()))

if __name__ == '__main__':
    train_data, dataX, dataY, char_to_int, int_to_char, chars = load_data(corpus, seq_length=seq_length, batch_size=batch_size)
    model = Net(len(chars), embedding_dim, hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(epochs):
        train(model, optimizer, epoch, train_data, log_interval)

        if (epoch + 1) % save_interval == 0:
            model.eval()
            torch.save(model, output_path)

    # Save mappings, vocabs, & model
    save_pickle((dataX, char_to_int, int_to_char, chars), output_c)

    model.eval()
    torch.save(model, output)
