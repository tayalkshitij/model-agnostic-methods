import numpy as np
import scipy as sp
import sklearn
import random
import time
import functools
from sklearn import preprocessing, model_selection
from keras.layers import Dropout,SpatialDropout1D,Conv1D,GlobalMaxPooling1D
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,ShuffleSplit
import re
from util import *
from model import *
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from callback_util import D2LCallback, LoggerCallback

from keras.callbacks import ModelCheckpoint

PATIENCE = 10
EPOCHS = 75
seed = 42
EPS_LSR_NOISY = 0.3

LOSS = [0,1,2,('crossentropy_reed_wrap_hard',0.3),('symmetric_cross_entropy_wrap',[2,1]),('lq_loss_wrap',0.3),'categorical_crossentropy',('crossentropy_reed_wrap_soft',0.3)]


# 0 - Label smoothing
# 1 - mixup
# 2 - dimensionality to learning

dataset_path = sys.argv[1]

query_label = pd.read_pickle(dataset_path)
DATASET = "".join(dataset_path.split('_')[0:2])
NOISE_RATIO = 0

print (f"Length of pandas dataframe is  {query_label.shape}")
labels_o = query_label["last_category"].to_list()
query = query_label["fastTextvectorencoding"].to_list()


def train_val_test(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.30,random_state=1)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.20,random_state=1)

    return train_x, train_y, val_x, val_y, test_x, test_y



for loss_function in LOSS:

    labels, encoder  = encodeLabel(labels_o)
    if loss_function == 0:
        labels = label_smoothing(labels,EPS_LSR_NOISY)
        loss_function = 'categorical_crossentropy'

    elif loss_function == 2:

        features = encodeTraining(query)
        train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels)
        model = get_model("DNN",input_tensor=None, input_shape=(128,), num_classes=train_y.shape[1])
        loss = lid_paced_loss()
        model.compile(loss=loss,optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

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


        model.fit_generator(text_generator_d2l(train_x, train_y),steps_per_epoch=features.shape[0]/128, epochs=EPOCHS,validation_data=(val_x,val_y),verbose=2,callbacks=callbacks)
        loss_v,ypred = model.evaluate(test_x,test_y,verbose=2)
        print (f"Accuracy for {loss_function}, neural network is {ypred}")
        continue


    features = encodeTraining(query)

    train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels)

    ypred = simple_DNN(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE,loss_function, EPOCHS)


    print (f"Accuracy for {loss_function}, neural network is {ypred}")



for loss_function in LOSS:


    labels, encoder  = encodeLabel(labels_o)

    query_label['clean'] = query_label["title"].apply(clean_text)
    query_label['clean'] = query_label['clean'].str.replace('\d+', '')

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 10
    # This is fixed.
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(query_label['clean'].values)

    X = tokenizer.texts_to_sequences(query_label['clean'].values)
    features = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)


    if loss_function == 0:
        labels = label_smoothing(labels,EPS_LSR_NOISY)
        loss_function = 'categorical_crossentropy'

    elif loss_function == 2:


        train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels)
        model = get_model("LSTM",input_tensor=None, input_shape=(10,), num_classes=train_y.shape[1])
        loss = lid_paced_loss()
        model.compile(loss=loss,optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

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

        model.fit_generator(text_generator_d2l(train_x, train_y),steps_per_epoch=features.shape[0]/128, epochs=EPOCHS,validation_data=(val_x,val_y),verbose=2,callbacks=callbacks)
        loss_v,ypred = model.evaluate(test_x,test_y,verbose=2)
        print (f"Accuracy for {loss_function}, LSTM is {ypred}")



        model = get_model("CNN",input_tensor=None, input_shape=(10,), num_classes=train_y.shape[1])
        model.compile(loss=loss,optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

        callbacks = []
        d2l_learning = D2LCallback(model, train_x, train_y, DATASET, NOISE_RATIO,
                                            epochs=EPOCHS,
                                            pace_type="D2L",
                                            init_epoch=init_epoch,
                                            epoch_win=epoch_win)

        callbacks.append(d2l_learning)
        model.fit_generator(text_generator_d2l(train_x, train_y),steps_per_epoch=features.shape[0]/128, epochs=EPOCHS,validation_data=(val_x,val_y),verbose=2,callbacks=callbacks)
        loss_v,ypred = model.evaluate(test_x,test_y,verbose=2)
        print (f"Accuracy for {loss_function}, CNN is {ypred}")
        continue



    train_x, train_y, val_x, val_y, test_x, test_y = train_val_test(features, labels)

    ypred = DL_LSTM(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE,loss_function, EPOCHS,MAX_NB_WORDS,EMBEDDING_DIM)

    print (f"Accuracy for {loss_function}, LSTM is {ypred}")

    ypred = DL_CNN(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE,loss_function, EPOCHS,MAX_NB_WORDS,EMBEDDING_DIM)

    print (f"Accuracy for {loss_function}, CNN is {ypred}")
