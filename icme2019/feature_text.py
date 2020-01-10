import pandas as pd
import numpy as np
import json
import lda
from collections import OrderedDict



def parsing_item_title_features():
    '''处理每个item对应的title词频'''

    print("process title features")

    item_dict = OrderedDict()

    # 保存 key 的编码
    key_dict = OrderedDict()
    n_keys = 0

    n_words = 0

    with open("data/track2_title.txt", 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            item_id = int(line['item_id'])
            if item_id not in item_dict:
                item_dict[item_id] = {}

            for key, value in line['title_features'].items():
                if key not in key_dict:
                    key_dict[key] = n_keys
                    n_keys += 1

                item_dict[item_id][key] = value
                n_words += value

    n_docs = len(item_dict)

    print(n_docs, n_keys, n_words)

    return item_dict, key_dict

def learn_topic(doc_dict, key_dict, n_topics):

    n_items = len(doc_dict)
    n_keys = len(key_dict)

    # X = np.zeros((n_items, n_keys), dtype=int)
    #
    # for iid, (item_id, doc) in enumerate(doc_dict.items()):
    #     for key, value in doc.items():
    #         X[iid, key_dict[key]] += value

    from sklearn.feature_extraction.text import CountVectorizer

    def split(s):
        return s.split(';')

    cv = CountVectorizer(analyzer=split)

    doc_string = []
    for iid, (item_id, doc) in enumerate(doc_dict.items()):
        ss = []
        for key, value in doc.items():
            for _ in range(value):
                ss.append(key)
        doc_string.append(';'.join(ss))

    X = cv.fit_transform(doc_string)

    lda_model = lda.LDA(n_topics=n_topics, n_iter=500, refresh=10)
    lda_model.fit(X)

    import matplotlib.pyplot as plt
    plt.plot(lda_model.loglikelihoods_[5:])

    lda_result = []
    for i, item_id in enumerate(doc_dict.keys()):
        feature = [item_id]
        topic_prop = lda_model.doc_topic_[i]
        feature.extend(topic_prop)
        lda_result.append(feature)

    out_df = pd.DataFrame(lda_result)
    names = ['item_id']

    for i in range(n_topics):
        names.append('lda_' + str(i))

    out_df.columns = names

    print(out_df.head())

    out_df.to_csv('data/feature_lda_' + str(n_topics) + '.csv', index=None)


if __name__ == "__main__":

    # txt 转换为 dict，方便特征拼接
    doc_dict, key_dict = parsing_item_title_features()

    learn_topic(doc_dict, key_dict, n_topics=8)


