import time
import math

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

    vocab = load_vocab('bookcorpus-vocab-truncated.pkl')
    ntokens = len(vocab.stoi)
    print(f'{ntokens} tokens in vocab')

    train_iterable = BookCorpusIterableDataset(train_dataset, vocab, batch_size=args.batch_size, sequence_length=args.sequence_length)
    train_loader = DataLoader(train_iterable, batch_size=None)
    val_iterable = BookCorpusIterableDataset(val_dataset, vocab, batch_size=args.batch_size, sequence_length=args.sequence_length)
    val_loader = DataLoader(val_iterable, batch_size=None)

    model = Transformer_Decoder(ntoken=ntokens, ninp=args.ninp, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    # Init tensorboard writer
    writer = SummaryWriter()

    best_val_loss = float("inf")
    epochs = args.epochs # The number of epochs
    best_model = None

    total_steps_in_dataset = 0
    total_steps_found = False
    nbatches = 'Unk'

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # train
        model.train()
        total_loss = 0.
        train_start_time = time.time()
        src_mask = model.generate_square_subsequent_mask(args.sequence_length).to(device)
        for i, (data, targets) in enumerate(train_loader):
            model.train()
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            # if data.size(0) != args.sequence_length:
            #     src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            # output = model(data, src_mask)
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = args.log_interval

            if not total_steps_found:
                total_steps_in_dataset = i+1
            else:
                nbatches = total_steps_in_dataset

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                cur_ppl = math.exp(cur_loss)
                elapsed = time.time() - train_start_time
                print(f'| epoch {epoch:3d} | {i:5d}/{nbatches} batches | lr {scheduler.get_lr()[0]:02.2f} '
                      f'| ms/batch {elapsed * 1000 / log_interval:5.2f} | loss {cur_loss:5.2f} | ppl {cur_ppl:8.2f}')
                total_loss = 0.
                train_start_time = time.time()


                # validate
                model.eval() # Turn on the evaluation mode
                total_loss = 0.
                src_mask = model.generate_square_subsequent_mask(args.sequence_length).to(device)
                with torch.no_grad():
                    for j, (data, targets) in enumerate(val_loader):
                        data, targets = data.to(device), targets.to(device)
                        # if data.size(0) != args.sequence_length:
                        #     src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                        output = model(data)
                        loss = criterion(output.view(-1, ntokens), targets)
                        total_loss += len(data) * loss.item()

                        if j + 1 == args.validation_steps:
                            break

                val_loss = total_loss / args.validation_steps
                val_ppl = math.exp(val_loss)
                print('-' * 89)
                print(f'| epoch {epoch:3d} | elapsed time: {time.time() - epoch_start_time:5.2f}s | '
                      f'val loss {val_loss:5.2f} | val ppl {val_ppl:8.2f}')
                print('-' * 89)

                total_loss = 0.

                if val_loss < best_val_loss:
                    print(f'Saving new best model: val loss improved from {best_val_loss:.3f} to {val_loss:.3f}')
                    best_val_loss = val_loss
                    best_model = model
                    # torch.save(best_model.state_dict(), f'checkpoints/net_epoch_{epoch}_step_{i}.pt')
                    torch.save(best_model, f'checkpoints/net_epoch_{epoch}_step_{i}.pt')

                steps_taken = (epoch-1) * total_steps_in_dataset + i
                writer.add_scalar('Loss/train', cur_loss, steps_taken)
                writer.add_scalar('Perplexity/train', cur_ppl, steps_taken)
                writer.add_scalar('Loss/val', val_loss, steps_taken)
                writer.add_scalar('Perplexity/val', val_ppl, steps_taken)

                # writer.flush()

        # went through the entire dataset once
        total_steps_found = True
        scheduler.step()


    print('training done')

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
