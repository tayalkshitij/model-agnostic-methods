import numpy as np
import scipy as sp
import sklearn
import random
import time
import functools
from sklearn import preprocessing, model_selection
from keras.layers import Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
import re
from util import *
from model import *
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from callback_util import D2LCallback, LoggerCallback
from keras.callbacks import ModelCheckpoint

import os
import numpy as np

PATIENCE = 10
EPOCHS = 15
seed = 42
EPS_LSR_NOISY = 0.3

LOSS = [0,1,2,('crossentropy_reed_wrap_hard',0.3),('symmetric_cross_entropy_wrap',[2,1]),('lq_loss_wrap',0.3),'categorical_crossentropy',('crossentropy_reed_wrap_soft',0.3)]

# 0 - Label smoothing
# 1 - mixup
# 2 - dimensionality to learning

dataset_path = sys.argv[1]




query_label = pd.read_pickle(dataset_path)

DATASET = "".join(dataset_path.split('_')[0:2])

NOISE_RATIO_LIST = [0.20]

print(f"Length of pandas dataframe is  {query_label.shape}")
labels_o = query_label["last_category"].to_list()
query = query_label["fastTextvectorencoding"].to_list()



def train_val_test(features, labels, NOISE_RATIO):

    if NOISE_RATIO == 0:
        train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.30, random_state=1)

        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.20, random_state=1)

        return train_x, train_y, val_x, val_y, test_x, test_y

    else:

        train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.30, random_state=1)

        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.20, random_state=1)

        ix_size = int(NOISE_RATIO * train_y.shape[0])

        ix = np.random.choice(train_y.shape[0], size=ix_size, replace=False)

        b = train_y[ix]

        np.random.shuffle(b)

        train_y[ix] = b

        return train_x, train_y, val_x, val_y, test_x, test_y



embeddings_index = {}

f = open(os.path.join('/home/kumarv/tayal007/Downloads/Projects/Noise_WWW/glove.42B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


def returnmatrix(matrix,embedding_matrix):
    p = np.zeros((matrix.shape[0],10,300))
    row_num = 0
    for row in matrix:
        for index,i in enumerate(row):
            p[row_num][index] = embedding_matrix[i]
        row_num = row_num + 1
    return p



for NOISE_RATIO in NOISE_RATIO_LIST:

    for loss_function in LOSS:

        labels, encoder = encodeLabel(labels_o)
        if loss_function == 0:
            labels = label_smoothing(labels, EPS_LSR_NOISY)
            loss_function = 'categorical_crossentropy'

        elif loss_function == 2:

            features = encodeTraining(query)
            train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels,NOISE_RATIO)
            model = get_model("DNN", input_tensor=None, input_shape=(128,), num_classes=train_y.shape[1])
            loss = lid_paced_loss()
            model.compile(loss=loss, optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

            callbacks = []
            init_epoch = 60
            epoch_win = 5
            d2l_learning = D2LCallback(model, train_x, train_y, DATASET, NOISE_RATIO,
                                       epochs=EPOCHS,
                                       pace_type="D2L",
                                       init_epoch=init_epoch,
                                       epoch_win=epoch_win)

            callbacks.append(d2l_learning)

            cp_callback = ModelCheckpoint("model/%s_%s_%s.hdf5" % ("D2L", DATASET, NOISE_RATIO),
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=False,
                                          save_weights_only=True,
                                          period=1)
            callbacks.append(cp_callback)

            model.fit_generator(text_generator_d2l(train_x, train_y), steps_per_epoch=features.shape[0] / 128,
                                epochs=EPOCHS, validation_data=(val_x, val_y), verbose=2, callbacks=callbacks)
            loss_v, ypred = model.evaluate(test_x, test_y, verbose=2)
            print(f"Accuracy for {loss_function}, neural network is {ypred}")
            continue

        features = encodeTraining(query)

        train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels,NOISE_RATIO)

        ypred = simple_DNN(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE, loss_function, EPOCHS)

        print(f"Accuracy for {loss_function}, neural network is {ypred}")

    for loss_function in LOSS:

        labels, encoder = encodeLabel(labels_o)

        query_label['clean'] = query_label["title"].apply(clean_text)
        query_label['clean'] = query_label['clean'].str.replace('\d+', '')

        # The maximum number of words to be used. (most frequent)
        MAX_NB_WORDS = 50000
        # Max number of words in each complaint.
        MAX_SEQUENCE_LENGTH = 10
        # This is fixed.
        EMBEDDING_DIM = 300

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(query_label['clean'].values)

        X = tokenizer.texts_to_sequences(query_label['clean'].values)
        features = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        if loss_function == 0:
            labels = label_smoothing(labels, EPS_LSR_NOISY)
            loss_function = 'categorical_crossentropy'

        elif loss_function == 2:

            train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels,NOISE_RATIO)

            word_index = tokenizer.word_index

            embedding_matrix = np.zeros((len(word_index) + 1, 300))
            for word, i in word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            train_x = returnmatrix(train_x, embedding_matrix)
            val_x = returnmatrix(val_x, embedding_matrix)
            test_x = returnmatrix(test_x, embedding_matrix)



            model = get_model("LSTM", input_tensor=None, input_shape=(10,300), num_classes=train_y.shape[1])
            loss = lid_paced_loss()
            model.compile(loss=loss, optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

            callbacks = []
            init_epoch = 60
            epoch_win = 5
            d2l_learning = D2LCallback(model, train_x, train_y, DATASET, NOISE_RATIO,
                                       epochs=EPOCHS,
                                       pace_type="D2L",
                                       init_epoch=init_epoch,
                                       epoch_win=epoch_win)

            callbacks.append(d2l_learning)

            cp_callback = ModelCheckpoint("model/%s_%s_%s.hdf5" % ("D2L", DATASET, NOISE_RATIO),
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=False,
                                          save_weights_only=True,
                                          period=1)
            callbacks.append(cp_callback)

            model.fit_generator(text_generator_d2l(train_x, train_y), steps_per_epoch=features.shape[0] / 128,
                                epochs=EPOCHS, validation_data=(val_x, val_y), verbose=2, callbacks=callbacks)
            loss_v, ypred = model.evaluate(test_x, test_y, verbose=2)
            print(f"Accuracy for {loss_function}, LSTM is {ypred}")

            model = get_model("CNN", input_tensor=None, input_shape=(10,300), num_classes=train_y.shape[1])
            model.compile(loss=loss, optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

            callbacks = []
            d2l_learning = D2LCallback(model, train_x, train_y, DATASET, NOISE_RATIO,
                                       epochs=EPOCHS,
                                       pace_type="D2L",
                                       init_epoch=init_epoch,
                                       epoch_win=epoch_win)

            callbacks.append(d2l_learning)
            model.fit_generator(text_generator_d2l(train_x, train_y), steps_per_epoch=features.shape[0] / 128,
                                epochs=EPOCHS, validation_data=(val_x, val_y), verbose=2, callbacks=callbacks)
            loss_v, ypred = model.evaluate(test_x, test_y, verbose=2)
            print(f"Accuracy for {loss_function}, CNN is {ypred}")
            continue

        train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels,NOISE_RATIO)

        word_index = tokenizer.word_index

        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        train_x = returnmatrix(train_x, embedding_matrix)
        val_x = returnmatrix(val_x, embedding_matrix)
        test_x = returnmatrix(test_x, embedding_matrix)

        ypred = DL_LSTM(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE, loss_function, EPOCHS, MAX_NB_WORDS,
                        EMBEDDING_DIM)

        print(f"Accuracy for {loss_function}, LSTM is {ypred}")

        ypred = DL_CNN(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE, loss_function, EPOCHS, MAX_NB_WORDS,
                       EMBEDDING_DIM)

        print(f"Accuracy for {loss_function}, CNN is {ypred}")
