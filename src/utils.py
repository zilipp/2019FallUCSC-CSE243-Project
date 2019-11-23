import os
from pathlib import Path
import sys
import logging
from logging import handlers
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit

_min_seq_len = 3  # Minimum length of sequence


def init_logger(log_file):
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file))

    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    output_format = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    std_out_handler = logging.StreamHandler(sys.stdout)
    std_out_handler.setFormatter(output_format)
    logging.getLogger().addHandler(std_out_handler)
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=(1048576*5), backupCount=7)
    file_handler.setFormatter(output_format)
    logging.getLogger().addHandler(file_handler)


def split_csv(file, data_dir, k, method='KFold'):
    train_df = pd.read_csv(file)

    logging.info("Cleaning data")
    # Fill "_na_" for title and author column
    train_df[['title', 'author']] = train_df[['title', 'author']].fillna(value='_na_')
    # Drop N/A for text column
    train_df = train_df.dropna()
    length = []
    [length.append(len(str(text))) for text in train_df['title']]
    train_df['length'] = length
    # dropping the outliers
    train_df = train_df.drop(train_df['title'][train_df['length'] < _min_seq_len].index, axis=0)

    logging.debug(train_df.groupby('length').size())

    logging.info("Splitting data")
    if method == 'KFold':
        kf = KFold(n_splits=k, shuffle=True, random_state=2018)
    elif method == 'StratifiedShuffleSplit':
        kf = StratifiedShuffleSplit(n_splits=k, random_state=2018)
    else:
        logging.error('split_csv input unknown method!')
        assert False

    logging.info("Saving kFold")
    i = 0
    for train_index, test_index in kf.split(train_df,
                                            y=train_df['label'] if method == 'StratifiedShuffleSplit' else None):
        test_df = train_df.iloc[test_index]
        test_df.to_csv((data_dir / (str(i) + 'csv')), encoding='utf-8', index=False)
        i += 1
