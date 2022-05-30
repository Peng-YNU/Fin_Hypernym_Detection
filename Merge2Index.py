import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pluralizer import Pluralizer
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM

from CustomizeDataset import CustomDataset

#
import matplotlib

matplotlib.use('Agg')

"""
p_type A: a/an x is a/an [MASK]
p_type B: xs are [MASK] #label word "swaps" is not covered in bert-base-uncased vocab#
p_type C: a/an x is a type of [MASK]
p_type D: a/an [MASK], such as a/an x
p_type E: a/an x is a/an [MASK], so is a/an x', x' is selected from the most nearest neighbors of x
"""


def parser_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--from_pretrained", type=str, default='bert-base-uncased')
    parser.add_argument("--data_dir", type=str, default='./data/FinSim-1/')
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument('--epoch', type=int, default=10, help='num of epochs')
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay rating")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')
    parser.add_argument("--max_seq_len", type=int, default=80, help="max input sentence length")
    parser.add_argument('--p_type', type=str, default='A', help="prompt type")
    parser.add_argument('--lm', type=str, default='BBBV', help='language model')

    args = parser.parse_args()

    return args


def is_vowel(char):
    all_vowels = 'aeiou'
    return char in all_vowels


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def load_data(path, name):
    with open(path + name) as rf:
        data = json.load(rf)
    return data


def construct(data, converter):
    pre_neighbors = pickle.load(open('./data/neighbor.pkl', 'rb'))
    inputs = []
    terms = []
    labels = []
    for item in data:

        term = item['term'].lower()
        label = item['label'].lower()

        # if label == 'corporate':
        #     continue

        if len(label.split(' ')) > 1:
            label = 'index'

        terms.append(term)
        prompt = ''
        s_term = converter.singular(term)
        s_label = converter.singular(label)

        p_term = converter.plural(term)
        p_label = converter.plural(label)

        if args.p_type == 'A':
            if is_vowel(s_term[0]) and is_vowel(s_label[0]):
                prompt = "an {} is an [MASK].".format(s_term)
            elif is_vowel(s_term[0]) and not is_vowel(s_label[0]):
                prompt = "an {} is a [MASK].".format(s_term)
            elif not is_vowel(s_term[0]) and is_vowel(s_label[0]):
                prompt = "a {} is an [MASK].".format(s_term)
            elif not is_vowel(s_term[0]) and not is_vowel(s_label[0]):
                prompt = "a {} is a [MASK].".format(s_term)
            labels.append(s_label)

        elif args.p_type == 'B':
            prompt = "{} are [MASK].".format(p_term)
            labels.append(p_label)

        elif args.p_type == 'C':
            if is_vowel(s_term[0]):
                prompt = "an {} is a type of [MASK].".format(s_term)
            else:
                prompt = "a {} is a type of [MASK].".format(s_term)
            labels.append(s_label)


        elif args.p_type == 'D':

            if is_vowel(s_term[0]) and is_vowel(s_label[0]):
                prompt = "an [MASK], such as an {}.".format(s_term)
            elif is_vowel(s_term[0]) and not is_vowel(s_label[0]):
                prompt = "a [MASK], such as an {}.".format(s_term)
            elif not is_vowel(s_term[0]) and is_vowel(s_label[0]):
                prompt = "an [MASK], such as a {}.".format(s_term)
            elif not is_vowel(s_term[0]) and not is_vowel(s_label[0]):
                prompt = "a [MASK], such as a {}.".format(s_term)

            labels.append(s_label)

        elif args.p_type == 'E':

            neig = []
            for w in s_term.split(' '):
                neig.append(pre_neighbors[w][2][1])
            co_term = ' '.join(neig).lower()
            if is_vowel(s_label[0]):
                if is_vowel(s_term[0]) and is_vowel(co_term[0]):
                    prompt = "an {} is an [MASK], so is an {}.".format(s_term, co_term)
                elif is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                    prompt = "an {} is an [MASK], so is a {}.".format(s_term, co_term)
                elif not is_vowel(s_term[0]) and is_vowel(co_term[0]):
                    prompt = "a {} is an [MASK], so is an {}.".format(s_term, co_term)
                elif not is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                    prompt = "a {} is an [MASK], so is a {}.".format(s_term, co_term)
            else:
                if is_vowel(s_term[0]) and is_vowel(co_term[0]):
                    prompt = "an {} is a [MASK], so is an {}.".format(s_term, co_term)
                elif is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                    prompt = "an {} is a [MASK], so is a {}.".format(s_term, co_term)
                elif not is_vowel(s_term[0]) and is_vowel(co_term[0]):
                    prompt = "a {} is a [MASK], so is an {}.".format(s_term, co_term)
                elif not is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                    prompt = "a {} is a [MASK], so is a {}.".format(s_term, co_term)

            labels.append(s_label)

        inputs.append(prompt)
    return inputs, labels, terms


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser_args()

    converter = Pluralizer()

    fs1_train_data = load_data(args.data_dir, 'train.json')
    fs1_test_data = load_data(args.data_dir, 'gold.json')
    fs2_data = load_data('./data/FinSim-2/', 'train.json')

    data = []
    fs1_train = 0
    fs1_test = 0
    fs2_one_count = 0
    fs2_two_count = 0
    fs2_count = 0

    for it in fs1_train_data:
        if it not in data:
            data.append(it)
            fs1_train += 1
    for it in fs1_test_data:
        if it not in data:
            data.append(it)
            fs1_test += 1

    print('FinSim-1 has {}'.format(len(data)))

    for it in fs2_data:

        if it not in data:
            if len(it['label'].split(' ')) == 2:
                fs2_two_count += 1
            if len(it['label'].split(' ')) == 1:
                fs2_one_count += 1
            data.append(it)
            fs2_count += 1

    print("in total {}".format(len(data)))

    print("FinSim-2 has {} unique one-word and {} two-word samples".format(fs2_one_count, fs2_two_count))

    inputs, labels, terms = construct(data, converter)

    model_path = "/mnt/disk/pengbo/workspace/pretrained_models/"
    model = None

    if args.lm == 'BBBV':
        tokenizer = BertTokenizer.from_pretrained(args.from_pretrained)
        model = BertForMaskedLM.from_pretrained(args.from_pretrained)
    elif args.lm == "FBBV":
        tokenizer = BertTokenizer.from_pretrained(args.from_pretrained)
        model = BertForMaskedLM.from_pretrained(model_path + "FinBERT-BaseVocab-Uncased")
    elif args.lm == "FBFV":
        tokenizer = BertTokenizer(vocab_file=model_path + "vocabs/FinVocab-Uncased.txt", do_lower_case=True)
        model = BertForMaskedLM.from_pretrained(model_path + "FinBERT-FinVocab-Uncased")

    model.to(device)

    LWs = set(labels)
    LW2id = dict()

    for lw in LWs:
        if lw not in tokenizer.vocab:
            tokenizer.add_tokens(lw)
            print('OOV token {} convert to:'.format(lw))
            print(tokenizer.convert_tokens_to_ids(lw))

        LW2id[lw] = tokenizer.convert_tokens_to_ids(lw)

    model.resize_token_embeddings(len(tokenizer))

    LW2id = {k: v for k, v in sorted(LW2id.items(), key=lambda item: item[1])}

    id2LW = {v: k for k, v in LW2id.items()}
    lw_ids = list(LW2id.values())

    encodings = tokenizer(inputs, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt',
                          max_length=80)
    LW_ids = tokenizer.convert_tokens_to_ids(labels)
    LW_ids = torch.tensor(LW_ids, dtype=torch.long)
    mask_coordinates = (encodings.input_ids == tokenizer.mask_token_id).long().nonzero()
    locs = torch.LongTensor(encodings.input_ids.shape[0]).fill_(-1)
    for i in range(mask_coordinates.shape[0]):
        coor = mask_coordinates[i]
        id = LW_ids[i]
        locs[i] = coor[1]
    encodings['locations'] = locs
    encodings['labels'] = LW_ids

    dataset = CustomDataset(encodings)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=8)

    predicts = []
    probabilities = []
    y_true = []

    true_hypers = []
    predict_hypers = []

    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            locations = batch['locations'].to(device)
            label_ids = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)['logits']
            for i in range(input_ids.shape[0]):
                loc = locations[i].item()
                y_true.append(lw_ids.index(label_ids[i].item()))
                true_hypers.append(id2LW[label_ids[i].item()])

                pred_prob = softmax(logits[i, loc, lw_ids])
                probabilities.append(pred_prob.detach().cpu().numpy())
                pred = pred_prob.argmax().detach().cpu().numpy().tolist()
                predicts.append(pred)
                predict_hypers.append(id2LW[lw_ids[pred]])

    acc = accuracy_score(y_true, predicts) * 100

    save_items = []
    top_k = 3
    total_rank = 0
    for i in range(len(probabilities)):
        y = y_true[i]
        prob = probabilities[i]
        Ranks = prob.argsort()[::-1].tolist()
        actual_r = Ranks.index(y)
        top_k_idx = Ranks[0:top_k]
        if y in top_k_idx:
            rank = top_k_idx.index(y) + 1
        else:
            rank = 4
        total_rank += rank

        save_items.append((terms[i], inputs[i], labels[i], actual_r + 1, [str(round(p, 2)) for p in prob.tolist()]))

    mean_rank = total_rank / len(probabilities)

    print('acc {}\t mean rank {}'.format(acc, mean_rank))
    #
    save_path = './save/FinSim-converted/Step_1_include/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '{}.{}.txt'.format(args.lm, args.p_type), 'w', encoding='utf-8') as wf:
        wf.write("Accuracy\tMean Rank\n")
        wf.write("{:.3f}\t{:.3f}\n".format(acc, mean_rank))

        wf.write('Term\tPrompt\tLabel\tRank\t{}\n'.format('\t'.join(list(LW2id.keys()))))
        for it in save_items:
            wf.write("{}\t{}\t{}\t{}\t{}\n".format(it[0], ''.join(it[1]), it[2], it[3], '\t'.join(it[4])))

    cm = confusion_matrix(true_hypers, predict_hypers, labels=list(LW2id.keys()))
    fig, ax = plt.subplots()

    sns.set(font_scale=0.8)  # Adjust to fit
    sns.heatmap(cm, annot=True, ax=ax, cmap="OrRd", fmt="g")
    label_font = {'size': '10'}  # Adjust to fit

    ax.set_xlabel('Prediction', fontdict=label_font)
    ax.set_ylabel('True', fontdict=label_font)

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.xaxis.set_ticklabels(list(LW2id.keys()), rotation=45)
    ax.yaxis.set_ticklabels(list(LW2id.keys()), rotation=45)

    save_fig_path = save_path + 'CM/'
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    plt.tight_layout()
    plt.savefig(save_fig_path + '{}.{}.svg'.format(args.lm, args.p_type), dpi='figure')
    # plt.show()
