# -*- coding: utf-8 -*-
import modin.pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys

sys.path.append('/home/tyc/yym/deep_ctr_torch')

from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, VectorFeat, get_feature_names
import torch
import torch.nn.functional as F
import torch.nn as nn


def get_feature_dict():
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did',
                       'video_duration', 'create_time', 'uid_cnt', 'did_cnt', 'item_id_cnt', 'author_id_cnt',
                       'uid_item_nunique', 'uid_author_nunique', 'item_uid_nunique', 'author_uid_nunique']

    dense_features = []
    social_graph_features = []
    user_click_features = []
    text_features = []
    video_audio_features = []

    for f1, f2 in [('uid', 'item_id'), ('uid', 'author_id')]:
        for i in range(8):
            # social_graph_features.append('dw_' + f1 + '_' + f2 + '_' + f1 + '_' + str(i))
            social_graph_features.append('dw_' + f1 + '_' + f2 + '_' + f2 + '_' + str(i))

    video_audio_features += ['vd' + str(i) for i in range(8)]

    user_click_features += ['w2v_item_id_' + str(i) for i in range(8)]

    text_features += ['lda_' + str(i) for i in range(8)]

    feature_dict = {'sparse_features': sparse_features,
                    'dense_features': dense_features,
                    'social_graph_features': social_graph_features,
                    'user_click_features': user_click_features,
                    'text_features': text_features,
                    'video_audio_features': video_audio_features}

    return feature_dict


if __name__ == "__main__":

    data = pd.read_csv('./data/sample_500w.csv')
    print(data.shape)
    print(data.head())

    feature_dict = get_feature_dict()

    target = ['like']

    # dense_features = feature_dict['social_graph_features'] + feature_dict['user_click_features'] + feature_dict[
    #     'text_features'] + feature_dict['video_audio_features']
    #
    # data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features

    # We have do this in preprocess.py

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in feature_dict['sparse_features'] if data[feat].nunique() < 100000] + \
                             [DenseFeat(feat, 1, ) for feat in feature_dict['dense_features']]

    # fixlen_feature_columns.append(VectorFeat('dpw', 16))
    # fixlen_feature_columns.append(VectorFeat('w2v', 8))
    # fixlen_feature_columns.append(VectorFeat('lda', 8))
    # fixlen_feature_columns.append(VectorFeat('vd', 8))

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, shuffle=False)

    train_model_input = {name: train[name] for name in feature_names if name in train}
    test_model_input = {name: test[name] for name in feature_names if name in test}

    # train_model_input['dpw'] = train[feature_dict['social_graph_features']].values
    # test_model_input['dpw'] = test[feature_dict['social_graph_features']].values
    #
    # train_model_input['w2v'] = train[feature_dict['user_click_features']].values
    # test_model_input['w2v'] = test[feature_dict['user_click_features']].values
    #
    # train_model_input['lda'] = train[feature_dict['text_features']].values
    # test_model_input['lda'] = test[feature_dict['text_features']].values
    #
    # train_model_input['vd'] = train[feature_dict['video_audio_features']].values
    # test_model_input['vd'] = test[feature_dict['video_audio_features']].values

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = PureFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   embedding_size=10,
                   task='binary', device=device)

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary',
    #                l2_reg_embedding=1e-5, device=device)

    # model = FiDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                 task='binary',
    #                 embedding_size=10,
    #                 dnn_hidden_units=(),
    #                 cin_layer_size=(40, 40,), cin_split_half=False, cin_activation=F.relu,
    #                 use_senet=False,
    #                 use_cosine_loss=False,
    #                 cin_flatten=True,
    #                 init_std=0.0001, seed=1024, dnn_dropout=0.5,
    #                 dnn_activation=F.relu, dnn_use_bn=False,
    #                 device=device)

    model.fit(train_model_input, train[target].values,
              batch_size=4096 * 32,
              epochs=40,
              lr=0.04,
              change_points=[20, ],
              factor=0.1,
              validation_data=[test_model_input, test[target].values],
              metrics=["auc"],
              shuffle=True,
              verbose=2)

    # model.print_best()

    # pred_ans = model.predict(test_model_input, 4096)

    # print("")
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

