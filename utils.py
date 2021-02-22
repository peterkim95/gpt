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

    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers per process (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--outf', type=str, default='generated.txt')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--words', type=int, default=30)
    parser.add_argument('--seed_string', type=str)

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
