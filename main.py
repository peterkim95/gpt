import shutil
import os
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from model import Transformer_Decoder, TransformerModel
from utils import load_vocab, get_args
from dataset import BookCorpusIterableDataset

best_val_loss = float("inf")

def main():
    args = get_args()

    vocab = load_vocab('bookcorpus-vocab-truncated.pkl')
    args.ntokens = len(vocab.stoi)

    # TODO: figure what the hell this is doing
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_val_loss
    args.gpu = gpu

    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = Transformer_Decoder(ntoken=args.ntokens, ninp=args.ninp, nhead=args.nhead,
                                nhid=args.nhid, nlayers=args.nlayers)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cudnn.benchmark = True

    # Data loading code goes here
    dataset = load_dataset("bookcorpus")['train'].train_test_split(train_size=0.8, test_size=0.2, shuffle=False, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    print('train val split done')

    vocab = load_vocab('bookcorpus-vocab-truncated.pkl')

    train_iterable_ds = BookCorpusIterableDataset(args.gpu, args.world_size, args.workers, train_dataset, vocab,
                                                  batch_size=args.batch_size, sequence_length=args.sequence_length)
    train_loader = DataLoader(train_iterable_ds, batch_size=None,
                              num_workers=args.workers, pin_memory=True)

    val_iterable_ds = BookCorpusIterableDataset(args.gpu, args.world_size, args.workers, val_dataset, vocab,
                                                batch_size=args.batch_size, sequence_length=args.sequence_length)
    val_loader = DataLoader(val_iterable_ds, batch_size=None, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Init tensorboard writer
    # writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_loss = validate(val_loader, model, criterion, args)

        # remember best val loss and save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


    # for epoch in range(1, args.epochs + 1):
    #     epoch_start_time = time.time()
    #
    #     # train
    #     model.train()
    #     total_loss = 0.
    #     train_start_time = time.time()
    #     for i, (data, targets) in enumerate(train_loader):
    #         model.train()
    #         if args.gpu is not None:
    #             data = data.cuda(args.gpu, non_blocking=True)
    #         if torch.cuda.is_available():
    #             targets = targets.cuda(args.gpu, non_blocking=True)
    #
    #         output = model(data)
    #         loss = criterion(output.view(-1, args.ntokens), targets)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #         log_interval = args.log_interval
    #
    #         if i % log_interval == 0 and i > 0:
    #             cur_loss = total_loss / log_interval
    #             cur_ppl = math.exp(cur_loss)
    #             elapsed = time.time() - train_start_time
    #
    #             nbatches = 'Unk'
    #             # if train_iterable.total_steps_found:
    #             #     nbatches = train_iterable.total_steps_in_dataset
    #
    #             # print(f'| epoch {epoch:3d} | {i:5d}/{nbatches} batches | lr {scheduler.get_lr()[0]:02.2f} '
    #             print(f'| epoch {epoch:3d} | {i:5d}/{nbatches} batches '
    #                   f'| ms/batch {elapsed * 1000 / log_interval:5.2f} | loss {cur_loss:5.2f} | ppl {cur_ppl:8.2f}')
    #             train_start_time = time.time()
    #
    #
    #             # validate
    #             model.eval() # Turn on the evaluation mode
    #             total_loss = 0.
    #             with torch.no_grad():
    #                 for j, (data, targets) in enumerate(val_loader):
    #                     if args.gpu is not None:
    #                         data = data.cuda(args.gpu, non_blocking=True)
    #                     if torch.cuda.is_available():
    #                         targets = targets.cuda(args.gpu, non_blocking=True)
    #                     # data, targets = data.to(device), targets.to(device)
    #                     output = model(data)
    #
    #                     loss = criterion(output.view(-1, args.ntokens), targets)
    #
    #                     total_loss += len(data) * loss.item()
    #
    #                     if j + 1 == args.validation_steps:
    #                         break
    #                     # if j % 5000 == 0:
    #                     #     print(j)
    #
    #             # val_iterable.setTotalStepsFound(True)
    #
    #             val_loss = total_loss / args.validation_steps
    #             # val_loss = total_loss / val_iterable.total_steps_in_dataset
    #             val_ppl = math.exp(val_loss)
    #             print('-' * 89)
    #             # print(f'| epoch {epoch:3d} | total val steps: {val_iterable.total_steps_in_dataset} | elapsed time: {time.time() - epoch_start_time:5.2f}s | '
    #             print(f'| epoch {epoch:3d} | total val steps: {123} | elapsed time: {time.time() - epoch_start_time:5.2f}s | '
    #                   f'val loss {val_loss:5.2f} | val ppl {val_ppl:8.2f}')
    #             print('-' * 89)
    #
    #             total_loss = 0.
    #
    #             if val_loss < best_val_loss:
    #                 print(f'Saving new best model: val loss improved from {best_val_loss:.3f} to {val_loss:.3f}')
    #                 best_val_loss = val_loss
    #                 best_model = model
    #                 # torch.save(best_model.state_dict(), f'checkpoints/net_epoch_{epoch}_step_{i}.pt')
    #                 torch.save(best_model, f'checkpoints/net_epoch_{epoch}_step_{i}.pt')
    #
    #             # steps_taken = (epoch-1) * train_iterable.total_steps_in_dataset + i
    #             # writer.add_scalar('Loss/train', cur_loss, steps_taken)
    #             # writer.add_scalar('Perplexity/train', cur_ppl, steps_taken)
    #             # writer.add_scalar('Loss/val', val_loss, steps_taken)
    #             # writer.add_scalar('Perplexity/val', val_ppl, steps_taken)
    #
    #             # writer.flush()
    #
    #             # scheduler.step()
    #
    #     # went through the entire dataset once
    #     # train_iterable.setTotalStepsFound(True)

    print('training done')
    # writer.flush()
    # writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        9999, # TODO: len(train_loader)
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (data, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            data = data.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        output = model(data)
        loss = criterion(output.view(-1, args.ntokens), targets)

        # record loss
        losses.update(loss.item(), data.size(1))

        # compute gradient and do SGD update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # TODO: necessary?
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        # len(val_loader),
        9999,
        [batch_time, losses],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        for i, (data, targets) in enumerate(val_loader):
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)

            output = model(data)
            loss = criterion(output.view(-1, args.ntokens), targets)

            losses.update(loss.item(), data.size(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return losses.avg


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == "__main__":
    main()
