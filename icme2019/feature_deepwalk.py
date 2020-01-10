import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec


def deepwalk(data, f1, f2, L):
    print("*" * 20)
    print("deepwalk: ", f1, f2)
    dic = {}

    # 构建二分图
    for item in data[[f1, f2]].values:
        try:
            dic['item_' + str(item[1])].add('user_' + str(item[0]))
        except:
            dic['item_' + str(item[1])] = {'user_' + str(item[0])}
        try:
            dic['user_' + str(item[0])].add('item_' + str(item[1]))
        except:
            dic['user_' + str(item[0])] = {'item_' + str(item[1])}

    print("=====creating=====")

    # 随机游走的序列长度
    path_length = 10
    sentences = []
    length = []
    for key in dic:
        sentence = [key]
        while len(sentence) != path_length:
            key = random.sample(dic[sentence[-1]], 1)[0]

            # 形成了环
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)

        # 每个生成的sentence的长度
        sentences.append(sentence)
        length.append(len(sentence))

        # 生成序列的进度
        if len(sentences) % 100000 == 0:
            print('generate seq: ', len(sentences))

    print('generate sentences num: ', len(sentences))
    print('sentence mean length: ', np.mean(length))

    print('training......')

    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, window=4, min_count=1, sg=1, workers=32, iter=20)
    print('outputing......')

    # 输出左侧节点的embedding
    values = set(data[f1].values)
    w2v = []
    for v in values:
        a = [v]
        a.extend(model['user_' + str(v)])
        w2v.append(a)
    out_df = pd.DataFrame(w2v)
    names = [f1]
    for i in range(L):
        names.append('dw_' + f1 + '_' + f2 + '_' + names[0] + '_' + str(i))

    out_df.columns = names
    print(out_df.head())
    out_df.to_csv('data/feature_deepwalk_' + f1 + '_' + f2 + '_' + f1 + '_' + str(L) + '.csv', index=None)

    # 输出右侧节点的embedding
    values = set(data[f2].values)
    w2v = []
    for v in values:
        a = [v]
        a.extend(model['item_' + str(v)])
        w2v.append(a)
    out_df = pd.DataFrame(w2v)
    names = [f2]
    for i in range(L):
        names.append('dw_' + f1 + '_' + f2 + '_' + names[0] + '_' + str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_csv('data/feature_deepwalk_' + f1 + '_' + f2 + '_' + f2 + '_' + str(L) + '.csv', index=None)


if __name__ == "__main__":
    data = pd.read_csv('data/final_track2_train.txt', sep='\t',
                       names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like',
                              'music_id', 'did', 'create_time', 'video_duration'])

    # deepwalk
    deepwalk(data, 'uid', 'item_id', 8)
    deepwalk(data, 'uid', 'author_id', 8)
