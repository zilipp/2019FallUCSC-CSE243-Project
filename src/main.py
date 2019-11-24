import os
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras
import shutil

# self defined module
from src.utils import split_csv, init_logger

# some config values
_user_logs_file = '..\\out\\logs\\user\\logs.txt'  # User logging directory.
_tf_logs_dir = '..\\out\\logs\\tf'  # TensorFlow logging directory.
_model_name = 'model.h5'  # Saved model name.
_model_dir = '..\\out\\checkpoints'  # Saved model directory
_data_dir = '..\\data'  # Data directory.
_mode = 'train'  # Whether to run 'train' or 'eval' or 'test' model.
_num_folds = 10  # Number of folds(k) used in cross validation.'
_eval_file = 5  # evaluation file in k-fold.
_num_gpus = 2  # Number of GPUs used.
_num_epochs = 50  # Number of epochs.
_batch_size = 2048
_eval_period = 1  # Number of epochs per evaluation.

_emb_size = 300  # Size of embedding vector
_vocab_size = 50000  # Number of unique words
_seq_length = 100  # Max length of sequence

if _num_gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def load_data():
    # check k-fold directory
    logging.info('Checking k-fold files')
    kfold_dir = os.path.join(_data_dir, 'kfold')
    if not os.path.exists(kfold_dir):
        os.mkdir(kfold_dir)
    if not os.listdir(kfold_dir):
        split_csv(os.path.join(_data_dir, 'train.csv'), kfold_dir, _num_folds, 'StratifiedShuffleSplit')

    # Load dataset
    logging.info('Loading train/test dataset')
    kfold_df = [pd.read_csv(os.path.join(kfold_dir, '{}.csv'.format(i))) for i in range(_num_folds)]
    test_df = pd.read_csv(os.path.join(_data_dir, 'test.csv'))
    logging.info('K-fold shape : {0} Test shape : {1}'.format(kfold_df[0].shape, test_df.shape))

    # split to train and val
    logging.info('Split to train and val')
    val_df = kfold_df[_eval_file]
    del kfold_df[_eval_file]
    train_df = pd.concat(kfold_df)
    logging.info('Train shape : {0} Eval shape : {1}'.format(train_df.shape, val_df.shape))

    # fill up the missing values
    logging.info('Fill up missing values')
    train_x = train_df['question_text'].fillna('_na_').values
    val_x = val_df['question_text'].fillna('_na_').values
    test_x = test_df['question_text'].fillna('_na_').values
    test_id = test_df['qid']

    # Get the target values
    logging.info('Get the target values')
    train_y = train_df['target'].values
    val_y = val_df['target'].values

    del train_df, val_df, test_df

    return train_x, train_y, val_x, val_y, test_x, test_id


def preprocess(train_x, val_x, test_x):
    # Tokenize the sentences
    logging.info('Tokenize sentences')
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=_vocab_size)
    tokenizer.fit_on_texts(list(train_x))
    train_x = tokenizer.texts_to_sequences(train_x)
    val_x = tokenizer.texts_to_sequences(val_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    # Pad the sentences
    logging.info('Pad sentences')
    train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=_seq_length)
    val_x = keras.preprocessing.sequence.pad_sequences(val_x, maxlen=_seq_length)
    test_x = keras.preprocessing.sequence.pad_sequences(test_x, maxlen=_seq_length)

    return train_x, val_x, test_x


def create_model():
    inp = keras.layers.Input(shape=(_seq_length,))
    embedded_seq = keras.layers.Embedding(_vocab_size, _emb_size)(inp)
    activations = keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(128, return_sequences=True))(
        embedded_seq)
    # compute importance for each step
    attention = keras.layers.Dense(1, activation='tanh')(activations)
    attention = keras.layers.Flatten()(attention)
    attention = keras.layers.Activation('softmax')(attention)
    attention = keras.layers.RepeatVector(256)(attention)
    attention = keras.layers.Permute([2, 1])(attention)
    multiply_layer = keras.layers.Multiply()
    x = multiply_layer([activations, attention])
    x = keras.layers.GlobalMaxPool1D()(x)
    x = keras.layers.Dense(16, activation='softmax')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(train_x, train_y, val_x, val_y):
    # Create a callback that saves weights only during training
    if os.path.exists(_tf_logs_dir):
        shutil.rmtree(_tf_logs_dir)
    os.mkdir(_tf_logs_dir)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=_tf_logs_dir)

    checkpoint_path = os.path.join(_model_dir, 'cp-{epoch:04d}.ckpt')
    if os.path.exists(_model_dir):
        shutil.rmtree(_model_dir)
    os.mkdir(_model_dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=_eval_period)
    # Setup model
    # https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_keras
    # https://www.tensorflow.org/tutorials/distribute/keras
    # https://www.tensorflow.org/api_docs/python/tf/distribute/ReductionToOneDevice
    # cross_device_ops = tf.distribute.ReductionToOneDevice(reduce_to_device='/device:CPU:0')
    # https://www.tensorflow.org/api_docs/python/tf/distribute/HierarchicalCopyAllReduce
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=2)
    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
    logging.info('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
    with mirrored_strategy.scope():
        logging.info('Setup model')
        model = create_model()
        # Save the weights using the `checkpoint_path` format
        logging.info(model.summary())

    # Train the model
    logging.info('Train the model')
    history = model.fit(train_x, train_y, batch_size=_batch_size, epochs=_num_epochs,
                        callbacks=[tensorboard, cp_callback], validation_data=(val_x, val_y))
    logging.info('Saving the model')
    model.save(os.path.join(_model_dir, _model_name))


def eval(val_x, val_y):
    logging.info('Trying to load latest checkpoint')
    latest = tf.train.latest_checkpoint(_model_dir)

    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    logging.info('Load weight into model')
    model.load_weights(latest)

    # Re-evaluate the model
    logging.info('Evaluate model')
    loss, acc = model.evaluate(val_x, val_y, verbose=1)
    logging.info('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    pred_noemb_val_y = model.predict([val_x], batch_size=_batch_size, verbose=1)
    for thresh in np.arange(0.1, 0.90, 0.01):
        thresh = np.round(thresh, 2)
        logging.info('F1 score at threshold {0} is {1}'.format(
            thresh, metrics.f1_score(val_y, (pred_noemb_val_y > thresh).astype(int))))


def test(val_x, val_y, test_x, test_id):
    model = keras.models.load_model(os.path.join(_model_dir, _model_name))

    # pred_noemb_val_y = model.predict([val_x], batch_size=1024, verbose=1)
    # for thresh in np.arange(0.1, 0.501, 0.01):
    #     thresh = np.round(thresh, 2)
    #     logging.info('F1 score at threshold {0} is {1}'.format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y > thresh).astype(int))))

    # Now let us get the test set predictions as well and save them
    logging.info('Get test set predictions and save them')
    pred_noemb_test_y = model.predict([test_x], batch_size=_batch_size, verbose=1)
    result = pd.DataFrame(
        {'qid': test_id.values.ravel(), 'prediction': (pred_noemb_test_y > 0.5).astype(int).ravel()},
        columns=['qid', 'prediction'])
    result.to_csv('submission.csv', index=False)


def main():
    init_logger(_user_logs_file)
    logging.info('======================start==========================')
    train_x, train_y, val_x, val_y, test_x, test_id = load_data()
    train_x, val_x, test_x = preprocess(train_x, val_x, test_x)
    train(train_x, train_y, val_x, val_y)
    eval(val_x, val_y)
    test(val_x, val_y, test_x, test_id)
    logging.info('Done!')


if __name__ == '__main__':
    main()
