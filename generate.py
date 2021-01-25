import torch

from utils import get_args, load_vocab

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    vocab = load_vocab('bookcorpus-vocab-truncated.pkl')
    ntokens = len(vocab.stoi)
    print(f'{ntokens} tokens in vocab')

    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    with open(args.outf, 'w+') as outf:
        with torch.no_grad():  # no tracking history
            for i in range(args.words):
                output = model(input, has_mask=False)
                # TODO: why move to cpu?
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)

                word = vocab.itos[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.generate_log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))

if __name__ == '__main__':
    main()