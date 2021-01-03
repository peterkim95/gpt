import time
import math

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from model import Transformer_Decoder
from utils import load_vocab, get_args, save_checkpoint
from dataset import BookCorpusIterableDataset

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("bookcorpus")['train'].train_test_split(train_size=0.8, test_size=0.2, shuffle=False, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    print('train val split done')

    vocab = load_vocab('bookcorpus-vocab.pkl')

    train_iterable = BookCorpusIterableDataset(train_dataset, vocab, batch_size=args.batch_size, sequence_length=args.sequence_length)
    train_loader = DataLoader(train_iterable, batch_size=None)
    val_iterable = BookCorpusIterableDataset(val_dataset, vocab, batch_size=args.batch_size, sequence_length=args.sequence_length)
    val_loader = DataLoader(val_iterable, batch_size=None)

    ntokens = len(vocab.stoi)
    model = Transformer_Decoder(ntoken=ntokens, ninp=args.ninp, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 3 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # train
        model.train()
        total_loss = 0.
        nbatches = 0
        train_start_time = time.time()
        src_mask = model.generate_square_subsequent_mask(args.sequence_length).to(device)
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            if data.size(0) != args.sequence_length:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 10
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - train_start_time
                # TODO: use nbatches here if known
                print('| epoch {:3d} | {:5d}/Unknown batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                train_start_time = time.time()

            nbatches = i + 1

        # validate
        model.eval() # Turn on the evaluation mode
        total_loss = 0.
        nbatches = 0
        src_mask = model.generate_square_subsequent_mask(args.sequence_length).to(device)
        with torch.no_grad():
            for i, (data, targets) in enumerate(val_loader):
                data, targets = data.to(device), targets.to(device)
                if data.size(0) != args.sequence_length:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
                nbatches = i + 1
        val_loss = total_loss / (nbatches - 1) # TODO: why minus 1?
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | '
              'val ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            save_checkpoint(best_model.state_dict(), f'checkpoints/net_epoch_{epoch}.pt')

        scheduler.step()


if __name__ == "__main__":
    main()
