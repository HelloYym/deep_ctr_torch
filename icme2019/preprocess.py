import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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

    data = pd.read_csv('./data/all_data.csv')
    print(data.shape)
    print(data.head())

    feature_dict = get_feature_dict()

    dense_features = feature_dict['social_graph_features'] + feature_dict['user_click_features'] + feature_dict[
        'text_features'] + feature_dict['video_audio_features']

    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv('./data/sample_500w.csv', index=None)
