import math
import torch
from torch import nn
import torch.nn.functional as func
from numpy.random import RandomState


def generate_square_subsequent_mask(total_len):
    mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
    mask = mask == 0  # lower -> False; others True
    return mask


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # d_model: word embedding size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        '''
        probably to prevent from rounding error
        e^(idx * (-log 10000 / d_model)) -> (e^(log 10000))^(- idx / d_model) -> 10000^(- idx / d_model) -> 1/(10000^(idx / d_model))
        since idx is an even number, it is equal to that in the formula
        '''
        pe[:, 0::2] = torch.sin(position * div_term)  # even number index, (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # odd number index
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, d_model) -> (1, max_len, d_model) -> (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # will not be updated by back-propagation, can be called via its name

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PETER_MF_R(nn.Module):
    def __init__(self, tgt_len, pad_idx, nuser, nitem, ntoken, emsize, nhead, nlayers, dropout=0.2):
        super(PETER_MF_R, self).__init__()
        self.user_embeddings = nn.Embedding(nuser,  emsize)
        self.item_embeddings = nn.Embedding(nitem,  emsize)
        self.rate_embeddings = nn.Embedding(6, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)

        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emsize, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=nlayers)
        self.hidden2token = nn.Linear(emsize, ntoken)

        self.src_len = 3
        self.emsize = emsize
        self.pad_idx = pad_idx
        self.attn_mask = generate_square_subsequent_mask(self.src_len + tgt_len)

        self.random_state = RandomState(1)
        self.user_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(nuser, emsize)).float()
        self.item_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(nitem, emsize)).float()
        self.rate_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(nitem, emsize)).float()

        self.init_weights()

    # 初始化权重
    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def set_parameter_requires_grad(self, rating=True, text=True):
        self.user_embeddings.requires_grad_(rating)
        self.item_embeddings.requires_grad_(rating)

        self.rate_embeddings.requires_grad_(text)
        self.word_embeddings.requires_grad_(text)
        self.encoder.requires_grad_(text)
        self.hidden2token.requires_grad_(text)

    def predict_rating(self, user, item):
        u_src = self.user_embeddings(user)
        i_src = self.item_embeddings(item)
        R_h = (u_src * i_src).sum(1)
        return R_h

    def predict_seq(self, user, item, rating, text):
        batch_size = user.size(0)
        tgt_len = text.size(1)
        device = user.device

        u_src = self.user_embeddings(user).unsqueeze(1)        # (batch_size, 1, emsize)
        i_src = self.item_embeddings(item).unsqueeze(1)        # (batch_size, 1, emsize)
        r_src = self.rate_embeddings(rating).unsqueeze(1)
        w_src = self.word_embeddings(text)                     # (batch_size, tgt_len, emsize)
        src = torch.cat(tensors=[u_src, i_src, r_src, w_src], dim=1)  # (batch_size, tgt_len + 3, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)

        left = torch.zeros(batch_size, self.src_len).bool().to(device)  # (batch_size, src_len)
        right = text == self.pad_idx
        attn_mask = self.attn_mask[:(self.src_len + tgt_len), :(self.src_len + tgt_len)].to(device)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        hidden = self.encoder(src, attn_mask, key_padding_mask)
        word_prob = self.hidden2token(hidden[:, self.src_len:])  # (batch_size, tgt_len, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, user, item, rating, text, rating_pred=True, seq_pred=True):
        if rating_pred:
            rating_p = self.predict_rating(user, item)            # (batch_size,)
        else:
            rating_p = None

        if seq_pred:
            log_word_prob = self.predict_seq(user, item, rating, text)  # (batch_size, tgt_len, ntoken)
        else:
            log_word_prob = None
        return rating_p, log_word_prob


class PETER_SCoR_R(nn.Module):
    def __init__(self, tgt_len, pad_idx, nuser, nitem, ntoken, emsize, nhead, nlayers, dropout=0.2, use_rating=False):
        super(PETER_SCoR_R, self).__init__()
        self.user_embeddings = nn.Embedding(nuser,  emsize)
        self.item_embeddings = nn.Embedding(nitem,  emsize)
        self.rate_embeddings = nn.Embedding(6, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)

        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emsize, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=nlayers)
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.hidden2rating = nn.Linear(1, 1)

        self.src_len = 3
        self.emsize = emsize
        self.pad_idx = pad_idx
        self.attn_mask = generate_square_subsequent_mask(self.src_len + tgt_len)

        self.random_state = RandomState(1)
        self.user_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(nuser, emsize)).float()
        self.item_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(nitem, emsize)).float()
        self.rate_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(nitem, emsize)).float()

        self.init_weights()

    # 初始化权重
    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def set_parameter_requires_grad(self, rating=True, text=True):
        self.user_embeddings.requires_grad_(rating)
        self.item_embeddings.requires_grad_(rating)

        self.rate_embeddings.requires_grad_(text)
        self.word_embeddings.requires_grad_(text)
        self.encoder.requires_grad_(text)
        self.hidden2token.requires_grad_(text)

    def predict_rating(self, user, item):
        u_src = self.user_embeddings(user)
        i_src = self.item_embeddings(item)
        p2 = (u_src - i_src).norm(p=2, dim=1).view(-1, 1)  # (batch_size,) -> (batch_size, 1)
        R_h = self.hidden2rating(p2).view(-1)
        return R_h

    def predict_seq(self, user, item, rating, text):
        batch_size = user.size(0)
        tgt_len = text.size(1)
        device = user.device

        u_src = self.user_embeddings(user).unsqueeze(1)        # (batch_size, 1, emsize)
        i_src = self.item_embeddings(item).unsqueeze(1)        # (batch_size, 1, emsize)
        r_src = self.rate_embeddings(rating).unsqueeze(1)
        w_src = self.word_embeddings(text)                     # (batch_size, tgt_len, emsize)
        src = torch.cat(tensors=[u_src, i_src, r_src, w_src], dim=1)  # (batch_size, tgt_len + 3, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)

        left = torch.zeros(batch_size, self.src_len).bool().to(device)  # (batch_size, src_len)
        right = text == self.pad_idx
        attn_mask = self.attn_mask[:(self.src_len + tgt_len), :(self.src_len + tgt_len)].to(device)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        hidden = self.encoder(src, attn_mask, key_padding_mask)
        word_prob = self.hidden2token(hidden[:, self.src_len:])  # (batch_size, tgt_len, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, user, item, rating, text, rating_pred=True, seq_pred=True):
        if rating_pred:
            rating_p = self.predict_rating(user, item)            # (batch_size,)
        else:
            rating_p = None

        if seq_pred:
            log_word_prob = self.predict_seq(user, item, rating, text)  # (batch_size, tgt_len, ntoken)
        else:
            log_word_prob = None
        return rating_p, log_word_prob