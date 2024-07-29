import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import Tuple, Optional
from torch import Tensor
from module import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, generate_square_subsequent_mask

class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):  # (batch_size, emsize)
        rating = torch.sum(user * item, dim=1)
        return rating

class SCoR(nn.Module):
    def __init__(self, maxR, minR):
        super(SCoR, self).__init__()
        self.maxR = maxR
        self.minR = minR

    def forward(self, user, item): # (batch_size, emsize)
        p2 = torch.norm(user - item, dim=1)
        rating = self.maxR - (self.maxR - self.minR) * p2 / 100
        return rating


class PETER_MF(nn.Module):
    def __init__(self, src_len, tgt_len, pad_idx, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, maxR=5, minR=0,
                 dropout=0.5):
        super(PETER_MF, self).__init__()
        self.pos_encoder = PositionalEncoding(emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid,
                                                 dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)

        self.hidden2token = nn.Linear(emsize, ntoken)
        self.recommender = MF()

        self.ui_len = 2
        self.src_len = src_len
        self.pad_idx = pad_idx
        self.emsize = emsize
        self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len)

        self.init_weights()

    # 初始化权重
    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, user, item):
        rating = self.recommender(user, item)  # (batch_size,)
        return rating

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, user, item, text, seq_prediction=True, context_prediction=True, rating_prediction=True):
        '''
        :param user: (batch_size,), torch.int64
        :param item: (batch_size,), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :param seq_prediction: bool
        :param context_prediction: bool
        :param rating_prediction: bool
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        '''
        device = user.device
        batch_size = user.size(0)
        total_len = self.ui_len + text.size(0)  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, self.ui_len).bool().to(device)  # (batch_size, ui_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        i_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, emsize)
        w_src = self.word_embeddings(text)  # (total_len - ui_len, batch_size, emsize)
        src = torch.cat([u_src, i_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, attns = self.transformer_encoder(src, attn_mask,
                                                 key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)

        if rating_prediction:
            rating = self.predict_rating(
                self.user_embeddings(user),
                self.item_embeddings(item)
            )  # (batch_size,)
        else:
            rating = None
        if context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)
        else:
            log_context_dis = None
        if seq_prediction:
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        return log_word_prob, log_context_dis, rating, attns

class PETER_SCoR(nn.Module):
    def __init__(self, src_len, tgt_len, pad_idx, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout=0.5, maxR=5, minR=0):
        super(PETER_SCoR, self).__init__()
        self.pos_encoder = PositionalEncoding(emsize, dropout)                  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)

        self.hidden2token = nn.Linear(emsize, ntoken)
        self.recommender = SCoR(maxR, minR)

        self.ui_len = 2
        self.src_len = src_len
        self.pad_idx = pad_idx
        self.emsize  = emsize
        self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len)

        self.init_weights()

    # 初始化权重
    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, user, item):
        rating = self.recommender(user, item)  # (batch_size,)
        return rating

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, user, item, text, seq_prediction=True, context_prediction=True, rating_prediction=True):
        '''
        :param user: (batch_size,), torch.int64
        :param item: (batch_size,), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :param seq_prediction: bool
        :param context_prediction: bool
        :param rating_prediction: bool
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        '''
        device = user.device
        batch_size = user.size(0)
        total_len = self.ui_len + text.size(0)  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, self.ui_len).bool().to(device)  # (batch_size, ui_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        i_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, emsize)
        w_src = self.word_embeddings(text)  # (total_len - ui_len, batch_size, emsize)
        src = torch.cat([u_src, i_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)

        if rating_prediction:
            rating = self.predict_rating(
                self.user_embeddings(user),
                self.item_embeddings(item)
            )  # (batch_size,)
        else:
            rating = None
        if context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)
        else:
            log_context_dis = None
        if seq_prediction:
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        return log_word_prob, log_context_dis, rating, attns
