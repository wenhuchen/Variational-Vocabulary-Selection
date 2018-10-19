#coding: utf-8
import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np
import math

triple_names = ["class", "title", "content"]
twin_names = ["class", "content"]

def ROC(y, x):
    x = preprocess(x)
    area = 0
    for xi, xj, yi, yj in zip(x[:-1], x[1:], y[:-1], y[1:]):
        area += (yi + yj) / 2 * abs(xj - xi)
    return area

def preprocess(x):
    #x = [math.log10(_) for _ in x]
    x = [_/max(x) for _ in x]
    return x

def get_train_path(dataset, step):
    """
    DB_TRAIN_PATH = "dbpedia_csv/train.csv"
    DB_TEST_PATH = "dbpedia_csv/test.csv"
    AG_TRAIN_PATH = "ag_news_csv/train.csv"
    AG_TEST_PATH = "ag_news_csv/test.csv"
    if dataset == "dbpedia" and step == 'train':
        return DB_TRAIN_PATH
    elif dataset == "dbpedia" and step == 'test':
        return DB_TEST_PATH
    elif dataset == "ag_news" and step == 'train':
        return AG_TRAIN_PATH
    elif dataset == "ag_news" and step == 'test':
        return AG_TEST_PATH
    """
    file_name = os.path.join("{}_csv/{}.csv".format(dataset, step))
    if os.path.exists(file_name):
        return file_name
    else:
        raise ValueError("Such dataset does not exist")

def download_dbpedia():
    dbpedia_url = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'

    wget.download(dbpedia_url)
    with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
        tar.extractall()

def clean_str(text):
    try:
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"@]", " ", text)
    except Exception:
        import pdb
        pdb.set_trace()
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    return text

def build_word_dict(dataset):
    dict_name = "vocab/{}_word_dict.pickle".format(dataset) 
    if not os.path.exists(dict_name):
        if "yelp" in dataset:
            train_df = pd.read_csv(get_train_path(dataset, 'train'), names=twin_names)
        else:
            train_df = pd.read_csv(get_train_path(dataset, 'train'), names=triple_names)

        contents = train_df["content"]

        words = list()
        for content in contents:
            if isinstance(content, str):
                for word in word_tokenize(clean_str(content)):
                    words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open(dict_name, "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open(dict_name, "rb") as f:
            word_dict = pickle.load(f)

    return word_dict

def build_word_dict_cutoff(dataset, cutoff=None, tokenize=True):
    dict_name = "vocab/{}_word_dict_cutoff{}.pickle".format(dataset, cutoff)
    if not os.path.exists(dict_name):
        if "yelp" in dataset:
            train_df = pd.read_csv(get_train_path(dataset, 'train'), names=twin_names)
        else:
            train_df = pd.read_csv(get_train_path(dataset, 'train'), names=triple_names)
        contents = train_df["content"]

        words = list()
        for content in contents:
            if isinstance(content, str):
                if tokenize:
                    for word in word_tokenize(clean_str(content)):
                        words.append(word)
                else:
                    for word in clean_str(content).split():
                        words.append(word)

        if cutoff is None:
            word_counter = collections.Counter(words).most_common()
        else:
            word_counter = collections.Counter(words).most_common(cutoff)

        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)
        with open(dict_name, "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open(dict_name, "rb") as f:
            word_dict = pickle.load(f)

    return word_dict

def revert(word_dict, sent):
    iword_dict = {y:x for x,y in word_dict.iteritems()}
    return " ".join([iword_dict[_] for _ in sent])

def build_word_dataset(dataset, step, word_dict, document_max_len, tokenize=False):
    if step == "train":
        if "yelp" in dataset:
            df = pd.read_csv(get_train_path(dataset, 'train'), names=twin_names)
        else:
            df = pd.read_csv(get_train_path(dataset, 'train'), names=triple_names)
        df = df.sample(frac=1)
    else:
        if "yelp" in dataset:
            df = pd.read_csv(get_train_path(dataset, 'test'), names=twin_names)
        else:
            df = pd.read_csv(get_train_path(dataset, 'test'), names=triple_names)
        #df = pd.read_csv(get_train_path(dataset, 'test'), names=["class", "title", "content"])

    # Shuffle dataframe
    if tokenize:
        x = list(map(lambda d: word_tokenize(clean_str(d)) if isinstance(d, str) else "empty line", df["content"]))
    else:
        x = list(map(lambda d: clean_str(d).split() if isinstance(d, str) else "empty line", df["content"]))        
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

    y = list(map(lambda d: int(d) - 1, list(df["class"])))

    return x, y


def build_char_dataset(dataset, step, model, document_max_len):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    if step == "train":
        if "yelp" in dataset:
            df = pd.read_csv(get_train_path(dataset, 'train'), names=twin_names)
        else:
            df = pd.read_csv(get_train_path(dataset, 'train'), names=triple_names)
        df = df.sample(frac=1)
    else:
        if "yelp" in dataset:
            df = pd.read_csv(get_train_path(dataset, 'test'), names=twin_names)
        else:
            df = pd.read_csv(get_train_path(dataset, 'test'), names=triple_names)

    # Shuffle dataframe
    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)

    alphabet_size = len(alphabet) + 2

    x = list(map(lambda content: list(map(lambda d: char_dict.get(d, char_dict["<unk>"]), content.lower())), df["content"]))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]], x))

    y = list(map(lambda d: int(d) - 1, list(df["class"])))

    return x, y, alphabet_size


def batch_iter(inputs, outputs, batch_size, num_epochs, test=True):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            x = inputs[start_index:end_index] 
            y = outputs[start_index:end_index]
            if test:
                yield epoch, x, y
            else:
                if len(x) < batch_size:
                    idx = np.random.choice(len(x), size=(batch_size, ))
                    x = x[idx]
                    y = y[idx]
                yield epoch, x, y
