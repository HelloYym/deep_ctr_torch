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

def convert(value, value_dict):

    if value_dict[str(value)] > 4:
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

    in_file = './data/raw_data_all.txt'
    out_file = './data/sample_all.csv'

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data = pd.read_csv(in_file)

    for dense_column in dense_features:
        data[dense_column] = data[dense_column].map(lambda value: itoc(value))

    target = ['label']
    features = sparse_features + dense_features

    for sparse_column in features:
        value_dict = defaultdict(lambda: 0)
        data[sparse_column].map(lambda value: count(value, value_dict))
        data[sparse_column] = data[sparse_column].map(lambda value: convert(value, value_dict))
        print('done ' + sparse_column)

    for feat in features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv(out_file, index=None)





