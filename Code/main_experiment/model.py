from keras.models import Sequential
from keras.utils import np_utils
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import keras
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.layers import BatchNormalization, Activation , Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,SpatialDropout1D,Conv1D,GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import Callback
from keras import optimizers
from numpy.random import permutation

import numpy as np

def crossentropy_reed_wrap_soft(_beta):
    def crossentropy_reed_core(y_true, y_pred):
        """
        This loss function is proposed in:
        Reed et al. "Training Deep Neural Networks on Noisy Labels with Bootstrapping", 2014
        :param y_true:
        :param y_pred:
        :return
        """

        # hyper param
        print(_beta)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
        # use predicted class proba directly to generate regression targets
        y_true_update = _beta * y_true + (1 - _beta) * y_pred

        # (2) compute loss as always
        _loss = -K.sum(y_true_update * K.log(y_pred), axis=-1)

        return _loss
    return crossentropy_reed_core


def crossentropy_reed_wrap_hard(beta):
    def crossentropy_reed_core(y_true, y_pred):
        """
        2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
        https://arxiv.org/abs/1412.6596
        :param y_true: 
        :param y_pred: 
        :return: 
        """
        print(beta)
      

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
        return -K.sum((beta * y_true + (1. - beta) * pred_labels) *
                      K.log(y_pred), axis=-1)
    
    return crossentropy_reed_core


def symmetric_cross_entropy_wrap(alpha, beta):
    """
    Symmetric Cross Entropy: 
    ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    https://arxiv.org/abs/1908.06112
    """
    def symmetric_cross_entropy(y_true, y_pred):
        
        print (f"Symmetric cross entropy with alpha {alpha} and beta {beta}")
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis = -1))
    return symmetric_cross_entropy

def symmetric_cross_entropy(alpha, beta):
    """
    Symmetric Cross Entropy: 
    ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    https://arxiv.org/abs/1908.06112
    """
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis = -1))
    return loss

def lid_paced_loss(alpha=1.0):
    """TO_DO
    Class wise lid pace learning, targeting classwise asymetric label noise.

    Args:      
      alpha: lid based adjustment paramter: this needs real-time update.
    Returns:
      Loss tensor of type float.
    """
    if alpha == 1.0:
        return symmetric_cross_entropy(alpha=0.1, beta=1.0)
    else:
        def loss(y_true, y_pred):
            pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
            y_new = alpha * y_true + (1. - alpha) * pred_labels
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            return -K.sum(y_new * K.log(y_pred), axis=-1)

        return loss


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
def lq_loss_wrap(_q):
    def lq_loss_core(y_true, y_pred):
        """
        This loss function is proposed in:
         Zhilu Zhang and Mert R. Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with
         Noisy Labels", 2018
        https://arxiv.org/pdf/1805.07836.pdf
        :param y_true:
        :param y_pred:
        :return:
        """

        # hyper param
        print(_q)

        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q

        return _loss
    return lq_loss_core

def crossentropy_max_wrap(_m):
    def crossentropy_max_core(y_true, y_pred):
        """
        This function is based on the one proposed in
        Il-Young Jeong and Hyungui Lim, "AUDIO TAGGING SYSTEM FOR DCASE 2018: FOCUSING ON LABEL NOISE,
         DATA AUGMENTATION AND ITS EFFICIENT LEARNING", Tech Report, DCASE 2018
        https://github.com/finejuly/dcase2018_task2_cochlearai
        :param y_true:
        :param y_pred:
        :return:
        """

        # hyper param
        print(_m)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        # threshold
        t_m = K.max(_loss) * _m
        _mask_m = 1 - (K.cast(K.greater(_loss, t_m), 'float32'))
        _loss = _loss * _mask_m

        return _loss
    return crossentropy_max_core


def crossentropy_outlier_wrap(_l):
    def crossentropy_outlier_core(y_true, y_pred):

        # hyper param
        print(_l)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median
            :param v:
            :return:
            """
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        # threshold
        t_l = _median_loss + _l*_std_loss
        _mask_l = 1 - (K.cast(K.greater(_loss, t_l), 'float32'))
        _loss = _loss * _mask_l

        return _loss
    return crossentropy_outlier_core

def loss_function(LOSS):
    # if LOSS[0] == 'crossentropy_reed_wrap':
    #         lossIs =  crossentropy_reed_wrap(float(LOSS[1]))
    if LOSS[0] == 'lq_loss_wrap':
            lossIs =  lq_loss_wrap(float(LOSS[1]))
    elif LOSS[0] == 'crossentropy_max_wrap':
            lossIs =  crossentropy_max_wrap(float(LOSS[1]))
    elif LOSS[0] == 'crossentropy_outlier_wrap':
            lossIs =  crossentropy_outlier_wrap(float(LOSS[1]))
    elif LOSS[0] == 'crossentropy_reed_wrap_hard':
            lossIs =  crossentropy_reed_wrap_hard(float(LOSS[1]))
    elif LOSS[0] == 'symmetric_cross_entropy_wrap':
            lossIs =  symmetric_cross_entropy_wrap(float(LOSS[1][0]),float(LOSS[1][1]))
    elif LOSS[0] == 'crossentropy_reed_wrap_soft':
            lossIs =  crossentropy_reed_wrap_soft(float(LOSS[1]))
    return lossIs

def simple_DNN(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE,LOSS, EPOCHS):

    
    model = Sequential()
    model.add(Dense(1024, input_dim = 128 ,activation = 'relu') )
    model.add(Dropout(0.2))
    model.add(Dense(512,activation = 'relu',))
    model.add(Dropout(0.2))
    model.add(Dense(train_y.shape[1], activation = 'softmax'))

    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE)]

    if LOSS != 1:

        if LOSS == 'categorical_crossentropy' or LOSS == 'mean_absolute_error':
            model.compile(optimizer=optimizers.Adam(0.001), loss=LOSS, metrics=['acc'])

        elif isinstance(LOSS, tuple):
            
            lossIs = loss_function(LOSS)
            model.compile(optimizer= optimizers.Adam(0.001), loss=lossIs, metrics=['acc'])
        
        model.fit(train_x, train_y,  validation_data=(val_x,val_y),  epochs=EPOCHS, callbacks=callbacks,verbose=2)
    
    else:
        model.compile(optimizer= optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['acc'])

        model.fit_generator(text_generator(train_x, train_y), steps_per_epoch = 200, nb_epoch = EPOCHS, verbose=2, validation_data=None)


    loss_v,ypred = model.evaluate(test_x,test_y,verbose=2)

    return ypred




def LR(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE,LOSS, EPOCHS):

    
    model = Sequential()
    model.add(Dense(train_y.shape[1], input_dim = train_x.shape[1] ,activation = 'softmax'))
    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE)]

    if LOSS != 1:

        if LOSS == 'categorical_crossentropy' or LOSS == 'mean_absolute_error':
            model.compile(optimizer=optimizers.Adam(0.001), loss=LOSS, metrics=['acc'])

        elif isinstance(LOSS, tuple):
            
            lossIs = loss_function(LOSS)
            model.compile(optimizer= optimizers.Adam(0.001), loss=lossIs, metrics=['acc'])
        
        model.fit(train_x, train_y,  validation_data=(val_x,val_y),  epochs=EPOCHS, callbacks=callbacks,verbose=2)
    
    else:
        model.compile(optimizer= optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['acc'])

        model.fit_generator(text_generator(train_x, train_y), steps_per_epoch = 200, nb_epoch = EPOCHS, verbose=2, validation_data=None)


    loss_v,ypred = model.evaluate(test_x,test_y,verbose=2)

    return ypred




def DL_LSTM(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE,LOSS, EPOCHS,MAX_NB_WORDS,EMBEDDING_DIM):

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=train_x.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(train_y.shape[1], activation='softmax'))


    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE)]

    if LOSS != 1:

        if LOSS == 'categorical_crossentropy' or LOSS == 'mean_absolute_error':
            model.compile(optimizer=optimizers.Adam(0.001), loss=LOSS, metrics=['acc'])

        elif isinstance(LOSS, tuple):
            
            lossIs = loss_function(LOSS)
            model.compile(optimizer= optimizers.Adam(0.001), loss=lossIs, metrics=['acc'])
        
        model.fit(train_x, train_y,  validation_data=(val_x,val_y),  epochs=EPOCHS, callbacks=callbacks,verbose=2)
    
    else:
        model.compile(optimizer= optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['acc'])

        model.fit_generator(text_generator(train_x, train_y), steps_per_epoch = 200, nb_epoch = EPOCHS, verbose=2, validation_data=None)


    loss_v,ypred = model.evaluate(test_x,test_y,verbose=2)

    return ypred

def DL_CNN(train_x, train_y, val_x, val_y, test_x, test_y, PATIENCE,LOSS, EPOCHS,MAX_NB_WORDS,EMBEDDING_DIM):

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=train_x.shape[1]))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(train_y.shape[1], activation='softmax'))



    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE)]

    if LOSS != 1:

        if LOSS == 'categorical_crossentropy' or LOSS == 'mean_absolute_error':
            model.compile(optimizer=optimizers.Adam(0.001), loss=LOSS, metrics=['acc'])

        elif isinstance(LOSS, tuple):
            
            lossIs = loss_function(LOSS)
            model.compile(optimizer= optimizers.Adam(0.001), loss=lossIs, metrics=['acc'])
        
        model.fit(train_x, train_y,  validation_data=(val_x,val_y),  epochs=EPOCHS, callbacks=callbacks,verbose=2)
    
    else:
        model.compile(optimizer= optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['acc'])

        model.fit_generator(text_generator(train_x, train_y), steps_per_epoch = 200, nb_epoch = EPOCHS, verbose=2, validation_data=None)


    loss_v,ypred = model.evaluate(test_x,test_y,verbose=2)

    return ypred



def text_generator(features, labels,batch_size = 1024):
    
    alpha = 0.2
    
    assert(features.shape[0] == labels.shape[0])

    no_datapoints = labels.shape[0]
    while True:
          # Select files (paths/indices) for the batch
            
            
        patch_ids_rand1 = permutation(no_datapoints)
         
        _features1 = features[patch_ids_rand1[0:batch_size]]
        _y_cat1 = labels[patch_ids_rand1[0:batch_size]]
        
        
                    
        patch_ids_rand2 = permutation(no_datapoints)
         
        _features2 = features[patch_ids_rand2[0:batch_size]]
        _y_cat2 = labels[patch_ids_rand2[0:batch_size]]
        
        
        # apply mixup, can be optmized, this is more readable
        y_cat_out = np.zeros_like(_y_cat1)
        _features = np.zeros_like(_features1)

        lam = np.random.beta(alpha, alpha, no_datapoints)

        for ii in range(batch_size):
            _features[ii] = lam[ii] * _features1[ii] + (1 - lam[ii]) * _features2[ii]
            y_cat_out[ii] = lam[ii] * _y_cat1[ii] + (1 - lam[ii]) * _y_cat2[ii]
        
        yield( _features, y_cat_out )


def text_generator_d2l(features, labels,batch_size = 1024):

      
        assert(features.shape[0] == labels.shape[0])

        no_datapoints = labels.shape[0]
        while True:
              # Select files (paths/indices) for the batch


            patch_ids_rand1 = permutation(no_datapoints)

            _features1 = features[patch_ids_rand1[0:batch_size]]
            _y_cat1 = labels[patch_ids_rand1[0:batch_size]]

            yield( _features1, _y_cat1 )





def get_model(model_name, input_tensor=None, input_shape=128, num_classes=10):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_shape):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if model_name == "DNN":

        x = Dense(1024, input_dim = 128)(img_input)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512)(x)
        x = Activation('relu',name='lid')(x)
        x = Dropout(0.2)(x)
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)


        return model

    elif model_name == "CNN":

    
        x = Embedding(50000, 100,input_length=input_shape)(img_input)
        x = Conv1D(128, 5)(x)
        x = Activation('relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(512,name='lid')(x)
        x = Activation('relu')(x)
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)
        return model


    elif model_name == "LSTM":

        x = Embedding(50000, 100,input_length=input_shape)(img_input)
        x = SpatialDropout1D(0.2)(x)
        x = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(x)
        x = Dense(512,name='lid')(x)
        x = Activation('relu')(x)
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)
        return model


    elif model_name == "LR":
        x = Dense(num_classes,name='lid')(img_input)
        x = Activation('softmax')(x)

        model = Model(img_input, x)
        return model

    return model