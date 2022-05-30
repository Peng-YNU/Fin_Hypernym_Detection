import fasttext
import json
from pluralizer import Pluralizer
import pickle

converter = Pluralizer()


def load_words(path, name):
    words = []
    with open(path + name) as rf:
        items = json.load(rf)
        for it in items:
            term = it['term'].lower().strip()
            s_term = converter.singular(term)
            p_term = converter.pluralize(term)
            for w in s_term.split(' '):
                words.append(w)
            for w in p_term.split(' '):
                words.append(w)
            label = it['label'].lower().strip()

            s_label = converter.singular(label)
            p_label = converter.pluralize(label)
            for w in s_label.split(' '):
                words.append(w)
            for w in p_label.split(' '):
                words.append(w)
    return words


FS_1_train_words = load_words('./data/FinSim-1/', 'train.json')
FS_1_test_words = load_words('./data/FinSim-1/', 'gold.json')

FS_2_train_words = load_words('./data/FinSim-2/', 'train.json')

Words = set(FS_1_train_words + FS_1_test_words + FS_2_train_words)

trained_model = fasttext.load_model("F:\Projects\pretrained_models\cc.en.300.bin")

neighbors = dict()
for w in Words:
    neighbors[w] = trained_model.get_nearest_neighbors(w)

with open('./data/neighbor.pkl', 'wb') as wf:
    pickle.dump(neighbors, wf)
