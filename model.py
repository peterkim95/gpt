import math
import copy

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

# TODO: Generalize. Assume num_gpus = 4, and num_layers = 12
class ModelParallelTransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(ModelParallelTransformerDecoder, self).__init__(decoder_layer, num_layers)
        mlist = [
            copy.deepcopy(decoder_layer).to('cuda:0'),
            copy.deepcopy(decoder_layer).to('cuda:0'),
            copy.deepcopy(decoder_layer).to('cuda:0'),

            copy.deepcopy(decoder_layer).to('cuda:1'),
            copy.deepcopy(decoder_layer).to('cuda:1'),
            copy.deepcopy(decoder_layer).to('cuda:1'),

            copy.deepcopy(decoder_layer).to('cuda:2'),
            copy.deepcopy(decoder_layer).to('cuda:2'),
            copy.deepcopy(decoder_layer).to('cuda:2'),

            copy.deepcopy(decoder_layer).to('cuda:3'),
            copy.deepcopy(decoder_layer).to('cuda:3'),
            copy.deepcopy(decoder_layer).to('cuda:3'),
        ]
        self.layers = nn.ModuleList(mlist)

        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        # output.to('cuda:0')
        # memory.to('cuda:0')
        # tgt_mask.to('cuda:0')
        # memory_mask.to('cuda:0')

        gpu_assignment = ['cuda:0', 'cuda:0', 'cuda:0', 'cuda:1', 'cuda:1', 'cuda:1', 'cuda:2', 'cuda:2', 'cuda:2', 'cuda:3', 'cuda:3', 'cuda:3']

        for mod, gpu in zip(self.layers, gpu_assignment):
            output.to(gpu)
            memory.to(gpu)
            tgt_mask.to(gpu)
            memory_mask.to(gpu)

            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            output.to(gpu)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Transformer_Decoder(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, model_parallel=False):
        super(Transformer_Decoder, self).__init__()
        self.model_type = 'Transformer-Decoder (T-D)'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layer = TransformerDecoderLayer(ninp, nhead, nhid, dropout, 'gelu')
        if model_parallel:
            self.transformer_decoder = ModelParallelTransformerDecoder(decoder_layer, nlayers)
        else:
            self.transformer_decoder = TransformerDecoder(decoder_layer, nlayers)
        self.embedding_encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.final_layer = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embedding_encoder.weight.data.uniform_(-initrange, initrange)
        self.final_layer.bias.data.zero_()
        self.final_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self.generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.embedding_encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=self.src_mask, memory_mask=self.src_mask)
        output = self.final_layer(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.src_mask = None

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self.generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
