import gc
import sklearn.decomposition as sk_decomposition
import pandas as pd
import numpy as np
import random

np.random.seed(2019)
random.seed(2019)


def embedding_pca(df, file, dim, columns):

        pca = sk_decomposition.PCA(n_components=dim, whiten=False, svd_solver='auto')
        pca.fit(df[columns])

        df_pca = pd.DataFrame(pca.transform(df[columns]))
        df_pca.columns = columns[:dim]
        df_pca['item_id'] = df['item_id'].values

        df_pca.to_csv(file, index=None)



if __name__ == "__main__":

    # 将音频特征文件转换为标准的csv
    # audio_feats = pd.read_json('data/track2_audio_features.txt', lines=True)
    # audio_feats.drop_duplicates(subset='item_id', inplace=True)
    #
    # audio_columns = ['ad' + str(i) for i in range(128)]
    # audio_df = pd.DataFrame(audio_feats.audio_feature_128_dim.tolist(), columns=audio_columns)
    # audio_df['item_id'] = audio_feats['item_id']
    # audio_df.fillna(0, inplace=True)
    # audio_df.to_csv('data/track2_audio_features_128.csv', index=False, float_format='%.4f')
    # embedding_pca(audio_df, 'data/feature_audio_16.csv', 16, audio_columns)
    # del audio_df
    # gc.collect()

    # 将视频特征文件转换为标准的csv
    video_feats = pd.read_json('data/track2_video_features.txt', lines=True)
    video_feats.drop_duplicates(subset='item_id', inplace=True)
    video_columns = ['vd' + str(i) for i in range(128)]
    video_df = pd.DataFrame(video_feats.video_feature_dim_128.tolist(), columns=video_columns)
    video_df['item_id'] = video_feats['item_id']
    video_df.fillna(0, inplace=True)
    video_df.to_csv('data/track2_video_features_128.csv', index=False, float_format='%.4f')
    embedding_pca(video_df, 'data/feature_video_8.csv', 8, video_columns)




