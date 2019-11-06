import modin.pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



if __name__ == "__main__":

    c = ['C' + str(i) for i in range(1, 27)]

    data = pd.read_csv('./data/sample_500w.csv')
    print(data.shape)
    print(data.head())

    print(data[c].max(axis=0))


