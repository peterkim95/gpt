import pickle
from argparse import ArgumentParser

import torch

def get_args():
    parser = ArgumentParser()

    parser.add_argument('--ninp', type=int, default=200)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--nhid', type=int, default=200)
    parser.add_argument('--nlayers', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sequence_length', type=int, default=35)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--validation_steps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5.0)

    args = parser.parse_args()
    return args


def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

def load_vocab(path):
    output = open(path, 'rb')
    vocab = pickle.load(output)
    output.close()
    return vocab
