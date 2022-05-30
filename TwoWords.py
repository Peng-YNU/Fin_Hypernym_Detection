import argparse
import json
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pluralizer import Pluralizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from CustomizeDataset import CustomDataset
from transformers import BertForMaskedLM

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
    parser.add_argument("--data_dir", type=str, default='./data/FinSim-2/')
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument('--epoch', type=int, default=10, help='num of epochs')
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay rating")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')
    parser.add_argument("--max_seq_len", type=int, default=80, help="max input sentence length")
    parser.add_argument('--p_type', type=str, default='B', help="prompt type")
    parser.add_argument('--lm', type=str, default='BBBV', help='language model')

    args = parser.parse_args()

    return args


def is_vowel(char):
    all_vowels = 'aeiou'
    return char in all_vowels


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def load_data(path, name, converter):
    with open(path + name) as rf:
        data = json.load(rf)
    pre_neighbors = pickle.load(open('./data/neighbor.pkl', 'rb'))
    inputs = []
    terms = []
    labels = []

    ind_inputs = []
    ind_labels = []

    hypers = []
    for item in data:

        term = item['term'].lower()
        label = item['label'].lower()
        if len(label.split(' ')) > 1:
            prompt_front = ''
            prompt_rear = ''

            s_term = converter.singular(term)
            p_term = converter.plural(term)

            lw_0, lw_1 = label.split(' ')[0], label.split(' ')[1]
            s_label = converter.singular(lw_1)
            p_label = converter.plural(lw_1)

            if args.p_type == 'A':
                if is_vowel(s_term[0]) and is_vowel(lw_0[0]):
                    prompt_front = "an {} is an [MASK] {}.".format(s_term, s_label)
                    prompt_rear = "an {} is an {} [MASK]".format(s_term, lw_0)
                elif is_vowel(s_term[0]) and not is_vowel(lw_0[0]):
                    prompt_front = "an {} is a [MASK] {}.".format(s_term, s_label)
                    prompt_rear = "an {} is a {} [MASK].".format(s_term, lw_0)
                elif not is_vowel(s_term[0]) and is_vowel(lw_0[0]):
                    prompt_front = "a {} is an [MASK] {}.".format(s_term, s_label)
                    prompt_rear = "a {} is an {} [MASK].".format(s_term, lw_0)
                elif not is_vowel(s_term[0]) and not is_vowel(label[0]):
                    prompt_front = "a {} is a [MASK] {}.".format(s_term, s_label)
                    prompt_rear = "a {} is a {} [MASK].".format(s_term, lw_0)


            elif args.p_type == 'B':
                prompt_front = "{} are [MASK] {}.".format(p_term, p_label)
                prompt_rear = "{} are {} [MASK].".format(p_term, lw_0)


            elif args.p_type == 'C':
                if is_vowel(s_term[0]):
                    prompt_front = "an {} is a type of [MASK] {}.".format(s_term, lw_1)
                    prompt_rear = "an {} is a type of {} [MASK].".format(s_term, lw_0)

                else:
                    prompt_front = "a {} is a type of [MASK] {}.".format(s_term, lw_1)
                    prompt_rear = "a {} is a type of {} [MASK].".format(s_term, lw_0)


            elif args.p_type == 'D':
                if is_vowel(s_term[0]) and is_vowel(lw_0[0]):
                    prompt_front = "an [MASK] {}, such as an {}.".format(s_label, s_term)
                    prompt_rear = "an {} [MASK], such as an {}.".format(lw_0, s_term)

                elif is_vowel(s_term[0]) and not is_vowel(lw_0[0]):
                    prompt_front = "a [MASK] {}, such as an {}.".format(s_label, s_term)
                    prompt_rear = "a {} [MASK], such as an {}.".format(lw_0, s_term)

                elif not is_vowel(s_term[0]) and is_vowel(lw_0[0]):
                    prompt_front = "an [MASK] {}, such as a {}.".format(s_label, s_term)
                    prompt_rear = "an {} [MASK], such as a {}.".format(lw_0, s_term)

                elif not is_vowel(s_term[0]) and not is_vowel(lw_0[0]):
                    prompt_front = "a [MASK] {}, such as a {}.".format(s_label, s_term)
                    prompt_rear = "a {} [MASK], such as a {}.".format(lw_0, s_term)


            elif args.p_type == 'E':
                neig = []
                for w in s_term.split(' '):
                    try:
                        neig.append(pre_neighbors[w][2][1])
                    except:
                        neig.append(w)

                co_term = ' '.join(neig).lower()
                if is_vowel(lw_0[0]):
                    if is_vowel(s_term[0]) and is_vowel(co_term[0]):
                        prompt_front = "an {} is an [MASK] {}, so is an {}.".format(s_term, s_label, co_term)
                        prompt_rear = "an {} is an {} [MASK], so is an {}.".format(s_term, lw_0, co_term)

                    elif is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                        prompt_front = "an {} is an [MASK] {}, so is a {}.".format(s_term, s_label, co_term)
                        prompt_rear = "an {} is an {} [MASK], so is a {}.".format(s_term, lw_0, co_term)

                    elif not is_vowel(s_term[0]) and is_vowel(co_term[0]):
                        prompt_front = "a {} is an [MASK] {}, so is an {}.".format(s_term, s_label, co_term)
                        prompt_rear = "a {} is an {} [MASK], so is an {}.".format(s_term, lw_0, co_term)

                    elif not is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                        prompt_front = "a {} is an [MASK] {}, so is a {}.".format(s_term, s_label, co_term)
                        prompt_rear = "a {} is an {} [MASK], so is a {}.".format(s_term, lw_0, co_term)

                else:
                    if is_vowel(s_term[0]) and is_vowel(co_term[0]):
                        prompt_front = "an {} is a [MASK] {}, so is an {}.".format(s_term, s_label, co_term)
                        prompt_rear = "an {} is a {} [MASK], so is an {}.".format(s_term, lw_0, co_term)

                    elif is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                        prompt_front = "an {} is a [MASK] {}, so is a {}.".format(s_term, s_label, co_term)
                        prompt_rear = "an {} is a {} [MASK], so is a {}.".format(s_term, lw_0, co_term)

                    elif not is_vowel(s_term[0]) and is_vowel(co_term[0]):
                        prompt_front = "a {} is a [MASK] {}, so is an {}.".format(s_term, s_label, co_term)
                        prompt_rear = "a {} is a {} [MASK], so is an {}.".format(s_term, lw_0, co_term)

                    elif not is_vowel(s_term[0]) and not is_vowel(co_term[0]):
                        prompt_front = "a {} is a [MASK] {}, so is a {}.".format(s_term, s_label, co_term)
                        prompt_rear = "a {} is a {} [MASK], so is a {}.".format(s_term, lw_0, co_term)

            if args.p_type == "B":
                terms.append(p_term)
                ind_labels.append(p_label)

            else:
                terms.append(s_term)
                ind_labels.append(s_label)

            labels.append(lw_0)
            inputs.append(prompt_front)
            ind_inputs.append(prompt_rear)
            hypers.append(label)
    return inputs, labels, terms, ind_inputs, ind_labels, hypers


def first_word(inputs, labels, terms, model, tokenizer):
    model.eval()
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
    ids = list(LW2id.values())

    encodings = tokenizer(inputs, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt',
                          max_length=80)
    LW_ids = tokenizer.convert_tokens_to_ids(labels)
    LW_ids = torch.tensor(LW_ids, dtype=torch.long)
    mask_coordinates = (encodings.input_ids == tokenizer.mask_token_id).long().nonzero()
    locs = torch.LongTensor(encodings.input_ids.shape[0]).fill_(-1)
    for i in range(mask_coordinates.shape[0]):
        coor = mask_coordinates[i]
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
                y_true.append(label_ids[i].item())

                pred_prob = softmax(logits[i, loc, ids])
                probabilities.append(pred_prob.detach().cpu().numpy())
                pred = pred_prob.argmax().detach().cpu().numpy().tolist()
                predicts.append(pred)
                predict_hypers.append(tokenizer.convert_ids_to_tokens(pred))

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
            rank = top_k_idx.index(y)
        else:
            rank = 4
        total_rank += rank

        save_items.append((terms[i], inputs[i], labels[i], actual_r + 1))

    mean_rank = total_rank / len(probabilities)

    print('acc {}\t mean rank {}'.format(acc, mean_rank))
    #
    save_path = './save/FinSim-Separate/First/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '{}.{}.txt'.format(args.lm, args.p_type), 'w', encoding='utf-8') as wf:
        wf.write('Term\tPrompt\tLabel\tRank\n')
        for it in save_items:
            wf.write("{}\t{}\t{}\t{}\n".format(it[0], ''.join(it[1]), it[2], it[3]))

        wf.write("Accuracy\tMean Rank\n")
        wf.write("{:.3f}\t{:.3f}\n".format(acc, mean_rank))

    return


def second_word(ind_inputs, ind_labels, terms, model, tokenizer):
    model.eval()
    LWs = set(ind_labels)
    LW2id = dict()

    for lw in LWs:
        if lw not in tokenizer.vocab:
            tokenizer.add_tokens(lw)
            print('OOV token {} convert to:'.format(lw))
            print(tokenizer.convert_tokens_to_ids(lw))

        LW2id[lw] = tokenizer.convert_tokens_to_ids(lw)
    model.resize_token_embeddings(len(tokenizer))

    LW2id = {k: v for k, v in sorted(LW2id.items(), key=lambda item: item[1])}

    predict_labels = []
    for w in LW2id.keys():
        if w in tokenizer.vocab:
            predict_labels.append(w)
        else:
            predict_labels.append('UNK')

    encodings = tokenizer(ind_inputs, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt',
                          max_length=80)
    LW_ids = tokenizer.convert_tokens_to_ids(ind_labels)
    LW_ids = torch.tensor(LW_ids, dtype=torch.long)
    mask_coordinates = (encodings.input_ids == tokenizer.mask_token_id).long().nonzero()
    locs = torch.LongTensor(encodings.input_ids.shape[0]).fill_(-1)
    for i in range(mask_coordinates.shape[0]):
        coor = mask_coordinates[i]
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
    ranks = []

    y_true = []

    true_hypers = []
    predict_words = []
    save_items = []

    softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            locations = batch['locations'].to(device)
            label_ids = batch['labels'].to(device)
            predictions = model(input_ids, attention_mask)[0]

            for i in range(input_ids.shape[0]):
                loc = locations[i].item()
                id = label_ids[i].item()

                y_true.append(id)
                probs = softmax(predictions[i, loc])
                # top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
                # for j, pred_idx in enumerate(top_k_indices):
                #     predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
                #     token_weight = top_k_weights[j]
                #     print("[MASK]: '%s'" % predicted_token, " | weights:", float(token_weight))

                r = torch.argsort(probs, descending=True)
                ranks.append((r == id).nonzero().item() + 1)
                probabilities.append(probs.detach().cpu().numpy())
                pred = probs.argmax().tolist()
                predicts.append(pred)
                predict_words.append(tokenizer.convert_ids_to_tokens(pred))

    equity_acc = 0
    credit_acc = 0
    e_num = 0
    c_num = 0
    acc = 0

    top_k = 3
    total_rank = 0

    for i in range(len(ranks)):

        if hypers[i].startswith('equity'):
            e_num += 1
            if ranks[i] == 1:
                equity_acc += 1
        elif hypers[i].startswith('credit'):
            c_num += 1
            if ranks[i] == 1:
                credit_acc += 1
        if ranks[i] == 1:
            acc += 1

    equity_acc = equity_acc / e_num * 100
    credit_acc = credit_acc / c_num * 100
    acc = acc / len(ranks) * 100

    for i in range(len(probabilities)):
        y = y_true[i]
        prob = probabilities[i]
        Ranks = prob.argsort()[::-1].tolist()
        top_k_idx = Ranks[0:top_k]
        if y in top_k_idx:
            rank = top_k_idx.index(y) + 1
        else:
            rank = 4
        total_rank += rank

        save_items.append((terms[i], ind_inputs[i], hypers[i], ranks[i]))

    mean_rank = total_rank / len(probabilities)
    save_path = './save/FinSim-Separate/SecondWord/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '{}.{}.txt'.format(args.lm, args.p_type), 'w', encoding='utf-8') as wf:
        wf.write("Accuracy\tMean Rank\n")
        wf.write("{:.3f}\t{:.3f}\n".format(acc, mean_rank))
        wf.write("Equity: \t Credit: \n")
        wf.write("{:.3f}\t{:.3f}\n".format(equity_acc, credit_acc))

        wf.write('Term\tPrompt\tLabel\tRank\n')
        for it in save_items:
            wf.write("{}\t{}\t{}\t{}\n".format(it[0], ''.join(it[1]), it[2], it[3]))

    return


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser_args()

    converter = Pluralizer()

    inputs, labels, terms, ind_inputs, ind_labels, hypers = load_data(args.data_dir, 'train.json', converter)

    model_path = "/mnt/disk/pengbo/workspace/pretrained_models/"

    if args.lm == 'BBBV':
        tokenizer = BertTokenizer.from_pretrained(args.from_pretrained)
        mask_model = BertForMaskedLM.from_pretrained(args.from_pretrained)
    elif args.lm == "FBBV":
        tokenizer = BertTokenizer.from_pretrained(args.from_pretrained)
        mask_model = BertForMaskedLM.from_pretrained(model_path + "FinBERT-BaseVocab-Uncased")
    elif args.lm == "FBFV":
        tokenizer = BertTokenizer(vocab_file=model_path + "vocabs/FinVocab-Uncased.txt", do_lower_case=True)
        mask_model = BertForMaskedLM.from_pretrained(model_path + "FinBERT-FinVocab-Uncased")

    mask_model.to(device)

    first_word(inputs, labels, terms, mask_model, tokenizer)

    second_word(ind_inputs, ind_labels, terms, mask_model, tokenizer)
