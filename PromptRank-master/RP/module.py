import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.init as initilize
import pickle
from numpy.random import RandomState


class PMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, is_sparse=False):
        super(PMF, self).__init__()
        self.n_users   = n_users
        self.n_items   = n_items
        self.n_factors = n_factors
        self.random_state = RandomState(1)

        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=is_sparse)

        self.user_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()
        self.item_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()

    def forward(self, users_index, items_index):
        user_h1 = self.user_embeddings(users_index)
        item_h1 = self.item_embeddings(items_index)
        R_h = (user_h1 * item_h1).sum(1)
        return R_h

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds


class SCoR(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super(SCoR, self).__init__()
        self.random_state = RandomState(1)

        self.user_embeddings = nn.Embedding(n_users, n_factors)  # 用户矩阵
        self.item_embeddings = nn.Embedding(n_items, n_factors)  # 物品矩阵
        self.linear = nn.Linear(1, 1)

        # 初始化权重
        initrange = 0.1
        self.user_embeddings.weight.data = torch.from_numpy(initrange * self.random_state.rand(n_users, n_factors)).float()
        self.item_embeddings.weight.data = torch.from_numpy(initrange * self.random_state.rand(n_items, n_factors)).float()

    def forward(self, user, item):                         # (batch_size, k)
        u_src = self.user_embeddings(user)                 # (batch_size, k)
        i_src = self.item_embeddings(item)                 # (batch_size, k)
        p2 = (u_src - i_src).norm(p=2, dim=1).view(-1, 1)  # (batch_size, k) -> (batch_size,) -> (batch_size, 1)
        rating = self.linear(p2).view(-1)                  # (batch_size, 1)
        return rating

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds


class SCoR2(nn.Module):
    def __init__(self, n_users, n_items, n_factors, linear_input=2):
        super(SCoR2, self).__init__()
        self.random_state = RandomState(1)
        self.linear_input = linear_input

        self.user_embeddings = nn.Embedding(n_users, n_factors)  # 用户矩阵
        self.item_embeddings = nn.Embedding(n_items, n_factors)  # 物品矩阵
        self.linear = nn.Linear(self.linear_input, 1)

        # 初始化权重
        initrange = 0.1
        self.user_embeddings.weight.data = torch.from_numpy(initrange * self.random_state.rand(n_users, n_factors)).float()
        self.item_embeddings.weight.data = torch.from_numpy(initrange * self.random_state.rand(n_items, n_factors)).float()

    def forward(self, user, item):                         # (batch_size, k)
        u_src = self.user_embeddings(user)                 # (batch_size, k)
        i_src = self.item_embeddings(item)                 # (batch_size, k)
        p2 = (u_src - i_src).norm(p=2, dim=1).view(-1, 1)  # (batch_size, k) -> (batch_size,) -> (batch_size, 1)
        mf = (u_src * i_src).sum(1).view(-1, 1)
        h1 = torch.cat([p2, mf], dim=1)
        rating = self.linear(h1).view(-1)
        return rating

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds


class LibMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, is_sparse=False):
        super(LibMF, self).__init__()
        self.n_users   = n_users
        self.n_items   = n_items
        self.n_factors = n_factors

        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=is_sparse)

        self.random_state = RandomState(1)
        self.user_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(n_users, n_factors)).float()
        self.item_embeddings.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(n_items, n_factors)).float()

    def forward(self, users_index, items_index):
        user_src = self.user_embeddings(users_index)
        item_src = self.item_embeddings(items_index)
        R_h = (user_src * item_src).sum(1)
        return R_h, user_src, item_src

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds


class SCoR_Plus(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super(SCoR_Plus, self).__init__()
        self.random_state = RandomState(1)

        self.user_embeddings = nn.Embedding(n_users, n_factors)  # 用户矩阵
        self.item_embeddings = nn.Embedding(n_items, n_factors)  # 物品矩阵
        self.linear = nn.Linear(1, 1)

        # 初始化权重
        initrange = 0.1
        self.user_embeddings.weight.data = torch.from_numpy(initrange * self.random_state.rand(n_users, n_factors)).float()
        self.item_embeddings.weight.data = torch.from_numpy(initrange * self.random_state.rand(n_items, n_factors)).float()

    def forward(self, user, item):                         # (batch_size, k)
        u_src = self.user_embeddings(user)                 # (batch_size, k)
        i_src = self.item_embeddings(item)                 # (batch_size, k)
        p2 = (u_src - i_src).norm(p=2, dim=1).view(-1, 1)  # (batch_size, k) -> (batch_size,) -> (batch_size, 1)
        rating = self.linear(p2).view(-1)                  # (batch_size, 1)
        return rating, u_src, i_src

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds