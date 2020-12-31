import pickle
from argparse import ArgumentParser

import torch

def get_args():
    parser = ArgumentParser()

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sequence_length', type=int, default=35)
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

def save_checkpoint(state, filename='checkpoints/checkpoint.pt'):
    torch.save(state, filename)