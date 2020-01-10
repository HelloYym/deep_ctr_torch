# import modin.pandas as pd
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

np.random.seed(2019)
random.seed(2019)


def norm(train_df, test_df, features):
    df = pd.concat([train_df, test_df])[features]
    scaler = preprocessing.QuantileTransformer(random_state=0)
    scaler.fit(df[features])
    train_df[features] = scaler.transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])


if __name__ == "__main__":

    data = pd.read_csv('data/feature_basic.csv')

    print(data.shape)

    dense_features = []
    social_graph_features = []
    user_click_features = []
    text_features = []
    video_audio_features = []

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did',
                       'video_duration', 'create_time', 'uid_cnt', 'did_cnt', 'item_id_cnt', 'author_id_cnt',
                       'uid_item_nunique', 'uid_author_nunique', 'item_uid_nunique', 'author_uid_nunique']

    # 拼接 deepwalk
    for f1, f2 in [('uid', 'item_id'), ('uid', 'author_id')]:
        # col = f1
        # df = pd.read_csv('data/feature_deepwalk_' + f1 + '_' + f2 + '_' + col + '_8.csv')
        # df = df.drop_duplicates([col])
        # data = pd.merge(data, df, on=col, how='left')
        # print(data.shape)

        col = f2
        df = pd.read_csv('data/feature_deepwalk_' + f1 + '_' + f2 + '_' + col + '_8.csv')
        df = df.drop_duplicates([col])
        data = pd.merge(data, df, on=col, how='left')
        print(data.shape)

    for f1, f2 in [('uid', 'item_id'), ('uid', 'author_id')]:
        for i in range(8):
            # social_graph_features.append('dw_' + f1 + '_' + f2 + '_' + f1 + '_' + str(i))
            social_graph_features.append('dw_' + f1 + '_' + f2 + '_' + f2 + '_' + str(i))

    print("done deepwalk!")

    # 拼接 video audio
    df = pd.read_csv('data/feature_video_8.csv')
    col = 'item_id'
    data = pd.merge(data, df, on=col, how='left')
    print(data.shape)

    # df = pd.read_csv('data/feature_audio_16.csv')
    # col = 'item_id'
    # data = pd.merge(data, df, on=col, how='left')
    # print(data.shape)

    video_audio_features += ['vd' + str(i) for i in range(8)]
    # video_audio_features += ['ad' + str(i) for i in range(16)]

    print("done video audio!")

    # 拼接 w2v

    df = pd.read_csv('data/feature_w2v_item_id_8.csv')
    data = pd.merge(data, df, on='item_id', how='left')
    print(data.shape)

    # df = pd.read_csv('data/feature_w2v_author_id_8.csv')
    # data = pd.merge(data, df, on='author_id', how='left')
    # print(data.shape)

    user_click_features += ['w2v_item_id_' + str(i) for i in range(8)]
    # user_click_features += ['w2v_author_id_' + str(i) for i in range(8)]

    print("done w2v!")

    # 拼接 lda

    df = pd.read_csv('data/feature_lda_8.csv')
    data = pd.merge(data, df, on='item_id', how='left')
    print(data.shape)

    text_features += ['lda_' + str(i) for i in range(8)]

    print("done lda!")

    # Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))

    if len(dense_features) > 0:
        data[dense_features] = mms.fit_transform(data[dense_features])

    data[social_graph_features] = mms.fit_transform(data[social_graph_features])
    data[user_click_features] = mms.fit_transform(data[user_click_features])
    data[video_audio_features] = mms.fit_transform(data[video_audio_features])

    if len(text_features) > 0:
        data[text_features] = mms.fit_transform(data[text_features])

    data.to_csv('data/all_data.csv', index=None)

    feature_dict = {'sparse_features': sparse_features,
                    'dense_features': dense_features,
                    'social_graph_features': social_graph_features,
                    'user_click_features': user_click_features,
                    'text_features': text_features,
                    'video_audio_features': video_audio_features}

    with open('data/all_data_feature_dict.pkl', 'wb') as pickle_file:
        pickle.dump(feature_dict, pickle_file)

    print("done!!!")
