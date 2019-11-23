# credit:
# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings

import os
from pathlib import Path
import logging
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

# self defined module
from src.utils import init_logger, split_csv

# get the root directory relative to current file instead of cwd
_root_dir = Path(os.path.dirname(os.path.abspath(__file__))) / '..'
_user_logs_file = _root_dir / Path('out/logs/user_logs/logs.txt') # User logging dire

# each word embedding to 300 dimension
embed_size = 300
# count of vocabulary words
max_features = 50000
# length of each sentences
max_len = 100


def load_data():
    logging.info('loading the dataset...')
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    logging.info('Train shape : ' + str(train_df.shape))
    logging.info('Test shape: ' + str(test_df.shape))
    logging.info('train_df.columns: ' + train_df.columns)
    logging.info('test_df.columns: ' + test_df.columns)
    count = train_df['question_text'].str.split().apply(len).value_counts()
    count.sort_index(inplace=True)
    count.index = count.index.astype(str) + ' words:'
    logging.info(count[-5:])
    return train_df, test_df


def split_train_val(train_df, test_df):
    logging.info('splitting train and val dataset...')
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=1)
    train_X = train_df["question_text"].fillna("_na_").values
    val_X = val_df["question_text"].fillna("_na_").values
    test_X = test_df["question_text"].fillna("_na_").values
    return train_df, val_df, train_X, val_X, test_X


def token_sentence(train_X, val_X, test_X ):
    logging.info('tokenizing sentence...')
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    logging.info('padding sentence...')
    train_X = pad_sequences(train_X, maxlen=max_len)
    val_X = pad_sequences(val_X, maxlen=max_len)
    test_X = pad_sequences(test_X, maxlen=max_len)
    return train_X, val_X, test_X


def build_model():
    logging.info('building the model...')
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    logging.info(model.summary())


def train_model():
    pass


def main():
    init_logger(_user_logs_file)
    logging.info('======================start==========================')
    train_df, test_df = load_data()

    # split train and valid
    train_df, val_df, train_X, val_X, test_X = split_train_val(train_df, test_df)
    train_y = train_df['target'].values
    val_y = val_df['target'].values

    # tokenizing and padding
    train_X, val_X, test_X = token_sentence(train_X, val_X, test_X)
    logging.info(train_df['target'].count())
    train_df.groupby('target').count()
    train_df['target'].plot.hist(bins=2, title='Distribution of label in trainning set')

    #build the model
    build_model()

    # train the model
    train_model()


if __name__ == '__main__':
    main()
