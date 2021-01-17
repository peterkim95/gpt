import pickle
from argparse import ArgumentParser

import torch

def get_args(print_args=True):
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
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--warmup_steps', type=int, default=2000)

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--outf', type=str, default='generated.txt')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--words', type=int, default=30)

    args = parser.parse_args()
    if print_args:
        print(args)
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
