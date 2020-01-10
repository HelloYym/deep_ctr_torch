import modin.pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



if __name__ == "__main__":


    data = pd.read_csv('./data/all_data.csv',)
    print(data.shape)
    print(data.head())

    sample_500w = data[:5000000]
    sample_100w = data[:1000000]

    sample_500w.to_csv('./data/raw_data_500w.txt', index=None)
    sample_100w.to_csv('./data/raw_data_100w.txt', index=None)
    data.to_csv('./data/raw_data_all.txt', index=None)

