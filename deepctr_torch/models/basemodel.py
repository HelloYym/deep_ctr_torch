# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""
from __future__ import print_function

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = self.create_embedding_matrix(self.sparse_feature_columns, 1, init_std, sparse=False)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
            linear_dense_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit = linear_sparse_logit + linear_dense_logit
        elif len(sparse_embedding_list) > 0:
            linear_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
        elif len(dense_value_list) > 0:
            linear_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
        return linear_logit

    def create_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        embedding_dict = nn.ModuleDict(
            {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=sparse) for feat in
             sparse_feature_columns}
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        return embedding_dict


class BaseModel(nn.Module):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, embedding_size=8,
                 l2_reg_embedding=1e-5, init_std=0.0001,
                 task='binary', device='cpu'):

        super(BaseModel, self).__init__()

        self.device = device  # device

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = self.create_embedding_matrix(dnn_feature_columns, embedding_size, init_std,
                                                           sparse=False)

        self.l2_reg_embedding = l2_reg_embedding

        self.out = PredictionLayer(task, )

    def get_loss(self, y, y_pred):

        loss = F.binary_cross_entropy(y_pred, y.squeeze())

        embedding_loss = self.get_regularization_loss(
            self.embedding_dict.parameters(), self.l2_reg_embedding)

        return loss + embedding_loss

    def reduce_lr(self, optimizer, epoch, factor=0.1, change_points=None):

        if change_points is not None and epoch in change_points:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * factor
                print(epoch, ": {0: .4f}".format(g['lr']))
            return True

    def fit(self, x=None,
            y=None,
            batch_size=4096,
            epochs=1,
            verbose=1,
            initial_epoch=0,
            validation_split=0.,
            validation_data=None,
            metrics=None,
            shuffle=True, ):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.

        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        if validation_data:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        # elif validation_split and 0. < validation_split < 1.:
        #     if hasattr(x[0], 'shape'):
        #         split_at = int(x[0].shape[0] * (1. - validation_split))
        #     else:
        #         split_at = int(len(x[0]) * (1. - validation_split))
        #     x, val_x = (slice_arrays(x, 0, split_at),
        #                 slice_arrays(x, split_at))
        #     y, val_y = (slice_arrays(y, 0, split_at),
        #                 slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        if isinstance(val_x, dict):
            val_x = [val_x[feature] for feature in self.feature_index]

        for i in range(len(val_x)):
            if len(val_x[i].shape) == 1:
                val_x[i] = np.expand_dims(val_x[i], axis=1)

        val_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(val_x, axis=-1)), torch.from_numpy(val_y))

        train_loader = DataLoader(dataset=train_tensor_data, shuffle=False, batch_size=batch_size,
                                  num_workers=8, pin_memory=False)

        val_loader = DataLoader(dataset=val_tensor_data, shuffle=False, batch_size=batch_size,
                                num_workers=8, pin_memory=False)

        model = self.cuda(self.device)
        model = nn.DataParallel(model, device_ids=[i for i in range(int(self.device.split(':')[1]), 8)])

        # 定义 优化方法
        optim = torch.optim.Adam(model.parameters(), lr=0.04)
        # optim = torch.optim.SGD(model.parameters(), lr=0.1)

        metrics = self._get_metrics(metrics)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}

            change_points = [10, 20, 30, 40, 50]
            self.reduce_lr(optim, epoch, factor=0.5, change_points=change_points)

            model.train()

            for index, (x_train, y_train) in enumerate(train_loader):

                x = Variable(x_train.cuda(self.device).float())
                y = Variable(y_train.cuda(self.device).float())

                y_pred = model(x).squeeze()

                optim.zero_grad()
                total_loss = self.get_loss(y, y_pred)

                total_loss_epoch += total_loss.item()
                total_loss.backward(retain_graph=True)
                optim.step()

                if verbose > 0 and epoch % verbose == 0:
                    for name, metric_fun in metrics.items():
                        if name not in train_result:
                            train_result[name] = []
                        train_result[name].append(metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy()))

            epoch_time = int(time.time() - start_time)

            if verbose > 0 and epoch % verbose == 0:

                model.eval()

                print('Epoch {0}/{1}'.format(epoch, epochs))

                # loss 与 metric 的区别，loss 包含 logloss 和 regloss
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, total_loss_epoch / steps_per_epoch)

                for name, result in train_result.items():
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(np.sum(result) / steps_per_epoch)

                if len(val_x) and len(val_y):
                    eval_result = self.evaluate(model, val_loader, metrics)

                    for name, result in eval_result.items():
                        eval_str += " - val_" + name + \
                                    ": {0: .4f}".format(result)
                print(eval_str)

    def evaluate(self, model, val_loader, metrics):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size:
        :return: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        """

        pred_ans = []
        eval_result = {}

        with torch.no_grad():
            for index, (x_test, y_test) in enumerate(val_loader):
                x = Variable(x_test.cuda(self.device).float())
                y = Variable(y_test.cuda(self.device).float())

                y_pred = model(x)
                pred_ans.append(y_pred)

                for name, metric_fun in metrics.items():
                    if name not in eval_result:
                        eval_result[name] = []
                    eval_result[name].append(metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy()))

        for key, eval_list in eval_result.items():
            eval_result[key] = np.mean(eval_list)

        return eval_result

    def predict(self, x, batch_size=4096):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))

        test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size,
                                 num_workers=16, pin_memory=False)

        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = Variable(x_test[0].cuda(self.device).float())

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans)

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        varlen_sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in varlen_sparse_feature_columns]

        varlen_sparse_embedding_list = list(
            map(lambda x: x.unsqueeze(dim=1), varlen_sparse_embedding_list))

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def create_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        embedding_dict = nn.ModuleDict(
            {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=sparse) for feat in
             sparse_feature_columns}
        )

        for feat in varlen_sparse_feature_columns:
            embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
                feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        return embedding_dict

    def compute_input_dim(self, feature_columns, embedding_size=1, include_sparse=True, include_dense=True,
                          feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = len(sparse_feature_columns) * embedding_size
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def get_regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = torch.zeros((1,)).cuda(self.device)
        for w in weight_list:
            if isinstance(w, tuple):
                l2_reg = torch.norm(w[1], p=p, )
            else:
                l2_reg = torch.norm(w, p=p, )
            reg_loss = reg_loss + l2_reg.cuda(self.device)
        reg_loss = weight_decay * reg_loss

        return reg_loss

    def _get_metrics(self, metrics):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
        return metrics_
