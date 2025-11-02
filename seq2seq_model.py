import random
import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext.data import Field
import spacy

# Tokenizers
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# Fields
german = Field(tokenize=tokenize_ger, lower=True,
               init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True,
                init_token="<sos>", eos_token="<eos>")

# Load Data
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english), root=".data"
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# Encoder/Decoder


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        # Combine bidirectional hidden states
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(
            self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))
        # Make hidden/cell match decoder n_layers
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = len(english.vocab)
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        enc_outputs, (hidden, cell) = self.encoder(src)
        # Repeat hidden/cell to match decoder layers
        if self.decoder.n_layers > 1:
            hidden = hidden.repeat(self.decoder.n_layers, 1, 1)
            cell = cell.repeat(self.decoder.n_layers, 1, 1)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs
