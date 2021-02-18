import torch
from tqdm import trange

from dataset import encode_raw_string
from utils import get_args, load_vocab

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f).module.to(device)
    model.eval()

    vocab = load_vocab('bookcorpus-vocab-truncated.pkl')
    ntokens = len(vocab.stoi)
    print(f'{ntokens} tokens in vocab')

    if args.seed_string is None:
        # choose random word
        random_word_index = torch.randint(ntokens, size=(1,))[0].item()
        seed_string = vocab.itos[random_word_index]
    else:
        seed_string = args.seed_string

    x = torch.unsqueeze(encode_raw_string(seed_string), dim=1).to(device)

    with open(args.outf, 'w+') as outf:
        outf.write(f'{seed_string}\n')
        with torch.no_grad():  # no tracking history
            for i in trange(args.words):
                output = model(x, has_mask=False)
                # TODO: why move to cpu?
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                x = torch.cat([x, word_tensor], 0)

                word = vocab.itos[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))


if __name__ == '__main__':
    main()
