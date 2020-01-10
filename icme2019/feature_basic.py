import pandas as pd
import numpy as np
import random

np.random.seed(2019)
random.seed(2019)



if __name__ == "__main__":

    data = pd.read_csv('data/final_track2_train.txt', sep='\t',
                       names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like',
                              'music_id', 'did', 'create_time', 'video_duration'])

    print(data.shape)

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', ]
    dense_features = []

    data['video_duration'] = pd.qcut(data['video_duration'], q=10, labels=False, duplicates='drop')
    sparse_features.append('video_duration')

    data['create_time'] = data['create_time'] % (24 * 3600) / 3600
    data['create_time'] = pd.qcut(data['create_time'], q=24, labels=False, duplicates='drop')
    sparse_features.append('create_time')  # 计数特征

    cols = ['uid', 'did', 'item_id', 'author_id']
    for c in cols:
        data[c + '_cnt'] = data[c].map(data[c].value_counts())
        data[c + '_cnt'] = pd.qcut(data[c + '_cnt'], q=10, labels=False, duplicates='drop')
        sparse_features.append(c + '_cnt')

    # ==============

    data['uid_item_nunique'] = data['uid'].map(data.groupby('uid')['item_id'].nunique())
    data['uid_item_nunique'] = pd.qcut(data['uid_item_nunique'], q=10, labels=False, duplicates='drop')
    sparse_features.append('uid_item_nunique')

    data['uid_author_nunique'] = data['uid'].map(data.groupby('uid')['author_id'].nunique())
    data['uid_author_nunique'] = pd.qcut(data['uid_author_nunique'], q=10, labels=False, duplicates='drop')
    sparse_features.append('uid_author_nunique')

    data['item_uid_nunique'] = data['item_id'].map(data.groupby('item_id')['uid'].nunique())
    data['item_uid_nunique'] = pd.qcut(data['item_uid_nunique'], q=30, labels=False, duplicates='drop')
    sparse_features.append('item_uid_nunique')

    data['author_uid_nunique'] = data['author_id'].map(data.groupby('author_id')['uid'].nunique())
    data['author_uid_nunique'] = pd.qcut(data['author_uid_nunique'], q=20, labels=False, duplicates='drop')
    sparse_features.append('author_uid_nunique')

    print('generate stats feats completed.')

    print(sparse_features)
    print(dense_features)

    data.to_csv('data/feature_basic.csv', index=None)

    print("*" * 80)
    print("done!")
