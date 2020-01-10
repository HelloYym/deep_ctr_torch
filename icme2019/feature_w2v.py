import pandas as pd
import random
from gensim.models import Word2Vec


def add_item_to_seq(dic, uid, item_id):
    try:
        dic[uid].append(str(item_id))
    except:
        dic[uid] = [str(item_id)]


def w2v(data, f, L):
    print("word2vec: ", f)
    sentence = []

    # 当前是按day写死的，之后可以考虑自定义区间
    for day in range(7):
        print('day: ', day)

        dic = {}
        data_day = data[data.day == day][['uid', f]]
        print(data_day.shape)
        data_day.apply(lambda row: add_item_to_seq(dic, row['uid'], row[f]), axis=1)
        print('user seqs cnt in this day: ', len(dic))

        sentence.extend(dic.values())
        dic.clear()

    print('total sentence: ', len(sentence))

    print('w2v training...')
    random.shuffle(sentence)

    # window: 当前词和之前几个词相关
    if f == 'item_id':
        model = Word2Vec(sentence, size=L, window=10, min_count=1, workers=32, iter=30)
    else:
        model = Word2Vec(sentence, size=L, window=10, min_count=1, workers=32, iter=10)

    print('outputing...')

    values = set(data[f].values)

    w2v = []
    for v in values:
        a = [v]
        a.extend(model[str(v)])
        w2v.append(a)

    out_df = pd.DataFrame(w2v)
    names = [f]

    for i in range(L):
        names.append('w2v_' + f + '_' + str(i))

    out_df.columns = names
    print(out_df.head())
    out_df.to_csv('data/feature_w2v_' + f + '_' + str(L) + '.csv', index=None)


if __name__ == "__main__":
    data = pd.read_csv('data/final_track2_train.txt', sep='\t',
                       names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like',
                              'music_id', 'did', 'create_time', 'video_duration'])

    print(data.shape)

    day = [i * 7 // len(data) for i in range(data.shape[0])]
    data['day'] = day

    w2v(data, 'author_id', 8)
    w2v(data, 'item_id', 8)
