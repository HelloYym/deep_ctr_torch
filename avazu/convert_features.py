import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys, csv, math
from collections import defaultdict
import random


def count(value, value_dict):
    value_dict[str(value)] += 1

def convert(value, value_dict, threshold=4):

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

    in_file = './data/raw_data_500w.txt'
    out_file = './data/sample_500w.csv'

    sparse_features = ['C' + str(i) for i in range(14, 22)]
    sparse_features += ['C1', 'hour', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                        'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']

    data = pd.read_csv(in_file)

    target = ['click']
    features = sparse_features

    for sparse_column in features:
        value_dict = defaultdict(lambda: 0)
        data[sparse_column].map(lambda value: count(value, value_dict))
        if sparse_column == 'device_ip':
            data[sparse_column] = data[sparse_column].map(lambda value: convert(value, value_dict, threshold=10))
        else:
            data[sparse_column] = data[sparse_column].map(lambda value: convert(value, value_dict))
        print('done ' + sparse_column)

    for feat in features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv(out_file, index=None)





