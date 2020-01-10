# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel, Linear
from ..inputs import combined_dnn_input
from ..layers import DNN, DAN, SENETLayer, FiDAN

import random
import numpy as np


class FiDeepFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns,
                 embedding_size=8,
                 dnn_hidden_units=(),
                 cin_layer_size=(10, 10,), cin_split_half=True, cin_activation=F.relu,
                 cin_flatten=False, use_senet=False, use_cosine_loss=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation=F.relu, dnn_use_bn=False, task='binary', device='cpu'):

        super(FiDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                       init_std=init_std,
                                       task=task, device=device)

        print('=====FiDeepFM=====')
        self.embedding_size = embedding_size

        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0

        self.cin_layer_size = cin_layer_size

        self.use_cin = True
        self.cin_flatten = cin_flatten
        self.use_senet = use_senet
        self.use_cosine_loss = use_cosine_loss

        field_num = len(self.embedding_dict) + len(self.vector_embedding_dict)

        # self.featuremap_num = sum(cin_layer_size) + field_num
        self.featuremap_num = sum(cin_layer_size)

        # self.cin = FiDAN(field_num, cin_layer_size,
        #                cin_activation, 3, seed, device=device)

        self.cin = DAN(field_num, cin_layer_size,
                       cin_activation, seed, device=device)

        if use_senet:
            cross_out_dim = self.featuremap_num * self.embedding_size * 2
        else:
            cross_out_dim = self.featuremap_num * self.embedding_size

        self.cin_linear = nn.Linear(cross_out_dim, 1, bias=True)
        self.bn = nn.BatchNorm1d(cross_out_dim)

        if self.use_dnn:
            self.dnn = DNN(cross_out_dim, dnn_hidden_units,
                           activation=dnn_activation, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)

        if use_senet:
            print("use senet")
            self.SE = SENETLayer(self.featuremap_num, self.featuremap_num // 3, seed, device)

        self._initialize_weights(init_std)

    def _initialize_weights(self, init_std):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, init_std)
                if m.bias is not None:
                    m.bias.data.zero_()

    # override
    def get_loss(self, y, y_pred, hidden_feature_maps):

        loss = super(FiDeepFM, self).get_loss(y, y_pred)

        if self.use_cosine_loss > 0:
            # cin_result = torch.cat(hidden_feature_maps, dim=1)
            # loss += self.calculate_cosineloss(cin_result) * 0.02
            for fmap in hidden_feature_maps:
                loss += self.calculate_cosineloss(fmap) * self.use_cosine_loss


        return loss

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict,
                                                                                  self.vector_embedding_dict)

        cin_input = torch.cat(sparse_embedding_list, dim=1)

        cin_feature_maps = self.cin(cin_input)[1:]
        # cin_feature_maps = [cin_input]

        # 在 feature map 维度，把每层的所有 feature map 进行拼接
        # shape : batch_size * n_feature_map * dim
        cin_result = torch.cat(cin_feature_maps, dim=1)

        if self.use_senet:
            se_cin_result = self.SE(cin_result)
            cin_result = torch.cat([se_cin_result, cin_result], dim=1)

        batch_size = cin_result.shape[0]

        # 把所有 feature map 进行 pooling
        # result shape : batch_size * n_feature_map
        cin_result = cin_result.reshape(batch_size, -1)
        cin_result = self.bn(cin_result)

        cin_logit = self.cin_linear(cin_result)

        if self.use_dnn:
            dnn_output = self.dnn(cin_result)
            dnn_logit = self.dnn_linear(dnn_output)

        if len(self.dnn_hidden_units) == 0:
            final_logit = cin_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:  # linear + DAN + Deep
            final_logit = cin_logit + dnn_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        if self.use_cosine_loss:
            return y_pred, cin_feature_maps
        else:
            return y_pred, None

    def calculate_cosineloss(self, feature_maps):

        maps = feature_maps

        batch_size = maps.size(0)
        num_maps = maps.size(1)
        # channel_num = int(num_maps / 2)
        # channel_num = num_maps
        channel_num = min(int(num_maps / 2), 20)
        eps = 1e-40
        random_seed = random.sample(range(num_maps), channel_num)
        maps = maps[:, random_seed, :]

        # X1 shape: batch_size * 1 * channel * dim
        X1 = maps.unsqueeze(1)
        # X1 shape: batch_size * channel * 1 * dim
        X2 = maps.unsqueeze(2)

        dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
        dist = dot12 / (torch.sqrt(dot11 * dot22 + eps))

        # tri_tensor = (
        #     (torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1] * channel_num))).expand(batch_size,
        #                                                                                                      channel_num,
        #                                                                                                      channel_num)).cuda(
        #
        #     self.device)
        #
        # dist_num = abs((tri_tensor * dist).sum(1).sum(1)).sum() / (batch_size * channel_num * (channel_num - 1) / 2)

        dist_num = abs((dist.sum(1).sum(1) - channel_num) / 2).sum() / (batch_size * channel_num * (channel_num - 1) / 2)

        return dist_num
