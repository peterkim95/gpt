import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from model import Transformer_Decoder
from utils import load_vocab, get_args
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

    # Init tensorboard writer
    writer = SummaryWriter()

    best_val_loss = float("inf")
    epochs = args.epochs # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # train
        # model.train()
        total_loss = 0.
        train_start_time = time.time()
        src_mask = model.generate_square_subsequent_mask(args.sequence_length).to(device)
        for i, (data, targets) in enumerate(train_loader):
            model.train()
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
            log_interval = args.log_interval
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                cur_ppl = math.exp(cur_loss)
                elapsed = time.time() - train_start_time
                # TODO: use nbatches here if known
                print('| epoch {:3d} | {:5d}/Unk batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, cur_ppl))
                total_loss = 0.
                train_start_time = time.time()


                # validate
                model.eval() # Turn on the evaluation mode
                total_loss = 0.
                src_mask = model.generate_square_subsequent_mask(args.sequence_length).to(device)
                with torch.no_grad():
                    for j, (data, targets) in enumerate(val_loader):
                        data, targets = data.to(device), targets.to(device)
                        if data.size(0) != args.sequence_length:
                            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                        output = model(data, src_mask)
                        # total_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()
                        loss = criterion(output.view(-1, ntokens), targets)
                        total_loss += loss.item()

                        if j + 1 == args.validation_steps:
                            break

                val_loss = total_loss / args.validation_steps
                val_ppl = math.exp(val_loss)
                print('-' * 89)
                # print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | '
                print('| epoch {:3d} | elapsed time: {:5.2f}s | val loss {:5.2f} | '
                      'val ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, val_ppl))
                print('-' * 89)

                total_loss = 0.

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    torch.save(best_model.state_dict(), f'checkpoints/net_epoch_{epoch}.pt')

                writer.add_scalar('Loss/train', cur_loss, i)
                writer.add_scalar('Perplexity/train', cur_ppl, i)
                writer.add_scalar('Loss/val', val_loss, i)
                writer.add_scalar('Perplexity/val', val_ppl, i)

        scheduler.step()

    print('training done')

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
