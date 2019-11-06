import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



if __name__ == "__main__":

    columns_name = ['label']
    columns_name += ['I' + str(i) for i in range(1, 14)]
    columns_name += ['C' + str(i) for i in range(1, 27)]

    data = pd.read_csv('./data/train.txt', sep='\t')
    data.columns = columns_name
    print(data.shape)
    print(data.head())

    data = data[:10000000]

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv('./data/sample_1000w.csv', index=None)

