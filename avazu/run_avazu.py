# -*- coding: utf-8 -*-
import modin.pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys

sys.path.append('/home/tyc/yym/deep_ctr_torch')

from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch
import torch.nn.functional as F
import torch.nn as nn

if __name__ == "__main__":

    data = pd.read_csv('./data/sample_500w.csv')
    print(data.shape)
    print(data.head())

    features = ['C' + str(i) for i in range(14, 22)]
    features += ['C1', 'hour', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                        'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']

    target = ['click']

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, shuffle=False)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = PureFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #            embedding_size=10,
    #            task='binary', device=device)

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                embedding_size=10, use_fm=True,
    #                dnn_hidden_units=(100, 100),
    #                l2_reg_linear=0.0001, l2_reg_embedding=0.0001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
    #                dnn_dropout=0.02,
    #                dnn_activation=F.relu, dnn_use_bn=True, task='binary', device=device)

    # model = DCN(dnn_feature_columns=dnn_feature_columns,
    #             embedding_size=10,
    #             cross_num=2,
    #             dnn_hidden_units=(100, 100),
    #             l2_reg_linear=0.0001, l2_reg_embedding=0.0001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
    #             dnn_dropout=0.02,
    #             dnn_activation=F.relu, dnn_use_bn=True, task='binary', device=device)

    model = FiDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary',
                    embedding_size=10,
                    dnn_hidden_units=(),
                    cin_layer_size=(40, 40, 40, 40), cin_split_half=False,
                    cin_activation=F.relu,
                    cin_flatten=True,
                    use_senet=True,
                    use_cosine_loss=0.01,
                    init_std=0.0001, seed=1024, dnn_dropout=0.05,
                    dnn_activation=F.relu, dnn_use_bn=True,
                    device=device)

    # model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                 task='binary',
    #                 embedding_size=10,
    #                 use_lr=False,
    #                 dnn_hidden_units=(),
    #                 cin_layer_size=(40, 40, 40, 40), cin_split_half=False, cin_activation=F.relu,
    #                 init_std=0.0001, seed=1024, dnn_dropout=0.02,
    #                 dnn_activation=F.relu, dnn_use_bn=True,
    #                 device=device)

    model.fit(train_model_input, train[target].values,
              batch_size=4096 * 16,
              epochs=30,
              lr=0.01,
              change_points=[5, 15],
              factor=0.1,
              validation_data=[test_model_input, test[target].values],
              metrics=["logloss", "auc"],
              shuffle=True,
              verbose=2)

    model.print_best()

    # pred_ans = model.predict(test_model_input, 4096)

    # print("")
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
