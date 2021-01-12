import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import IterableDataset, DataLoader

from utils import load_vocab

class BookCorpusIterableDataset(IterableDataset):

    def __init__(self, dataset, vocab, batch_size, sequence_length):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')

        self.dataset.set_format() # set get item return format into python object

        self.total_steps_in_dataset = 0
        self.total_steps_found = False

    def __iter__(self):
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

    def encode(self, example):
        # print(example)
        # print(torch.tensor([self.vocab[token] for token in self.tokenizer(example['text'])], dtype=torch.long))
        return torch.tensor([self.vocab[token] for token in self.tokenizer(example['text'])], dtype=torch.long)

    def setTotalStepsFound(self, b):
        self.total_steps_found = b

def main():
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