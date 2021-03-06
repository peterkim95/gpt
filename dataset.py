import os
from io import open

import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import IterableDataset, DataLoader

from utils import load_vocab

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class BookCorpusIterableDataset(IterableDataset):

    def __init__(self, metadata, gpu_index, world_size, num_workers, dataset, vocab, batch_size, sequence_length, steps_limit=100):
        self.metadata = metadata
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')

        self.dataset.set_format() # set get item return format into python object

        self.total_steps_in_dataset = 0
        self.total_steps_found = False

        self.main_process_index = 0 if gpu_index is None else gpu_index
        self.workers = num_workers
        self.total_workers = num_workers * world_size
        self.gpu_index = gpu_index
        self.steps_limit = steps_limit

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            token_stream = torch.LongTensor()
            total_batch_size = self.batch_size * (self.sequence_length + 1)
            for example in self.dataset:
                token_stream = torch.cat([token_stream, self.encode(example)])
                if token_stream.numel() >= total_batch_size:
                    batch, token_stream = token_stream[:total_batch_size], token_stream[total_batch_size:]
                    batch = batch.view(self.batch_size, self.sequence_length + 1).t()
                    x, y = batch[:self.sequence_length, :], batch[1:, :].reshape(-1)

                    if not self.total_steps_found:
                        self.total_steps_in_dataset += 1
                    yield x, y

        else: # in a worker process
            # split workload
            worker_id = worker_info.id
            worker_index = (self.workers * self.main_process_index) + worker_id
            token_stream = torch.LongTensor()
            total_batch_size = self.batch_size * (self.sequence_length + 1)
            
            print(f'{self.gpu_index}: {self.metadata} starting worker {worker_index}')

            for example in self.dataset.shard(num_shards=self.total_workers, index=worker_index, contiguous=True):
                token_stream = torch.cat([token_stream, self.encode(example)])
                if token_stream.numel() >= total_batch_size:
                    batch, token_stream = token_stream[:total_batch_size], token_stream[total_batch_size:]
                    batch = batch.view(self.batch_size, self.sequence_length + 1).t()
                    x, y = batch[:self.sequence_length, :], batch[1:, :].reshape(-1)

                    if not self.total_steps_found:
                        self.total_steps_in_dataset += 1
                        # print(f'p={self.gpu_index}, w={worker_index}: {self.steps_limit}, {self.total_steps_in_dataset}')
                        if self.total_steps_in_dataset >= self.steps_limit:
                            # print(f'{self.gpu_index}: breaking')
                            break
                    yield x, y

    def encode(self, example):
        return torch.tensor([self.vocab[token] for token in self.tokenizer(example['text'])], dtype=torch.long)

    def setTotalStepsFound(self, b):
        self.total_steps_found = b


def encode_raw_string(s):
    vocab = load_vocab('bookcorpus-vocab-truncated.pkl')
    tokenizer = get_tokenizer('basic_english')
    return torch.tensor([vocab[token] for token in tokenizer(s)], dtype=torch.long)


def main():
    # debug code for dataloader
    dataset = load_dataset("bookcorpus")['train'].train_test_split(train_size=0.8, test_size=0.2, shuffle=False, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    print('train val split done')

    vocab = load_vocab('bookcorpus-vocab.pkl')

    train_iterable = BookCorpusIterableDataset(train_dataset, vocab, batch_size=20, sequence_length=35)
    train_loader = DataLoader(train_iterable, batch_size=None)
    for x, y in train_loader:
        print(x, y)
        exit()


if __name__ == "__main__":
    main()
