import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys, csv, math
from collections import defaultdict
import random

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



def count(value, value_dict):
    value_dict[str(value)] += 1

def convert(value, value_dict, threshold=3):

    if value_dict[str(value)] > threshold:
        return str(value)
    else:
        return str('#' + str(value_dict[str(value)]))

def itoc(val):
    if pd.isnull(val):
        return 'nan'

    val = int(val)
    if val > 2:
        val = int(math.log(float(val)) ** 2)
    else:
        val = 'sp' + str(val)

    return str(val)

if __name__ == "__main__":

    # in_file = './data/raw_data_all.txt'
    out_file = './data/sample_all.csv'
    in_file = out_file

    feature_dict = get_feature_dict()

    # sparse_features = feature_dict['sparse_features']

    data = pd.read_csv(in_file)

    # features = sparse_features
    #
    # for sparse_column in features:
    #     value_dict = defaultdict(lambda: 0)
    #     data[sparse_column].map(lambda value: count(value, value_dict))
    #     data[sparse_column] = data[sparse_column].map(lambda value: convert(value, value_dict))
    #     print('done ' + sparse_column)

    dense_features = feature_dict['social_graph_features'] + feature_dict['user_click_features'] + feature_dict[
        'text_features'] + feature_dict['video_audio_features']
    data[dense_features] = data[dense_features].fillna(0, )

    # for feat in features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])

    # data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv(out_file, index=None)





