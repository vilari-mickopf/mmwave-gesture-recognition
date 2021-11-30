#! /usr/bin/env python

import os
import gc

import numpy as np

#  Disable tensorflow logs
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import callbacks

from mmwave.data.formats import GESTURE

import colorama
from colorama import Fore
colorama.init(autoreset=True)

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

# Init tf gpu
def set_tensorflow_config(per_process_gpu_memory_fraction=1):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    config.gpu_options.allow_growth=True
    tf.compat.v1.Session(config=config)
    tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
set_tensorflow_config()


class Model:
    def __init__(self, num_of_frames=50, num_of_objs=65, num_of_data_in_obj=5,
                 num_of_classes=9):
        self.model = None
        self.model_file = os.path.join(os.path.dirname(__file__), '.model')

        self.num_of_frames = num_of_frames
        self.num_of_objs = num_of_objs
        self.num_of_data_in_obj = num_of_data_in_obj
        self.num_of_classes = num_of_classes

        self.frame_size = self.num_of_objs*self.num_of_data_in_obj

    def __padd_data(self, data):
        padded_data = []

        # Pad objects
        zero_obj = [0.]*self.num_of_data_in_obj
        for sample in data:
            if len(sample) > self.num_of_objs:
                sample = sample[:self.num_of_objs]

            padded_sample = preprocessing.sequence.pad_sequences(
                sample, maxlen=self.num_of_objs, dtype='float32',
                padding='post', value=zero_obj)

            padded_data.append(padded_sample)

        # Pad frames
        zero_frame = [zero_obj for _ in range(self.num_of_objs)]
        padded_data = preprocessing.sequence.pad_sequences(
            padded_data, maxlen=self.num_of_frames, dtype='float32',
            padding='post', value=zero_frame)

        return np.asarray(padded_data)

    def __prep_data(self, X, y=None):
        # X has been already normalized while importing the data from .csv files
        X = self.__padd_data(X)
        X = X.reshape((len(X), self.num_of_frames, self.frame_size))

        if y is not None:
            y = utils.to_categorical(y)
            return X, y

        return X

    def create_model(self):
        pass

    def train(self, X_train, y_train, X_val, y_val):
        X_train, y_train = self.__prep_data(X_train, y_train)
        X_val, y_val = self.__prep_data(X_val, y_val)

        self.create_model()
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=1000,
                       validation_data=(X_val, y_val),
                       callbacks=[callbacks.EarlyStopping(patience=100,
                                                          restore_best_weights=True),
                                  callbacks.ModelCheckpoint(self.model_file,
                                                            verbose=True,
                                                            save_best_only=True)])

    def load(self):
        # Clear old model from memory
        del self.model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        print('Loading model...', end='')
        self.model = tf.keras.models.load_model(self.model_file)
        print(f'{Fore.GREEN}Done.')

    def evaluate(self, X, y):
        if self.model is None:
            print(f'{Fore.RED}Model not created.')
            return

        X, y = self.__prep_data(X, y)
        preds = self.model.evaluate(X, y)
        print(f'Loss: {round(preds[0], 4)}', end=' ')
        print(f'Acc: {round(preds[1], 4)}')

    def predict(self, X, debug=False):
        if self.model is None:
            print(f'{Fore.RED}Model not created.')
            return

        y_pred = self.model.predict(self.__prep_data(X))
        best_guess = [y_pred[0].tolist().index(x) for x in sorted(y_pred[0], reverse=True)]
        best_value = sorted(y_pred[0], reverse=True)

        if debug:
            for guess, val in zip(best_guess, best_value):
                print(f'{Fore.YELLOW}Best guess: {GESTURE.to_str(guess)}: {val:.2f}')
            print(f'{Fore.CYAN}------------------------------\n')

        if best_value[0] >= .9:
            print(f'{Fore.GREEN}Gesture recognized: ', end='')
            print(Fore.BLUE + GESTURE.to_str(best_guess[0]))
            print(f'{Fore.CYAN}==============================\n')


class ConvModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file = os.path.join(os.path.dirname(__file__), '.conv_model')

    def create_model(self):
        self.model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.num_of_frames, self.frame_size)),

            layers.Conv1D(128, kernel_size=3),
            layers.Conv1D(128, kernel_size=3),
            layers.ReLU(),
            layers.Dropout(.5),
            layers.MaxPooling1D(),

            layers.Conv1D(256, kernel_size=3),
            layers.ReLU(),
            layers.Dropout(.5),
            layers.MaxPooling1D(),

            layers.Conv1D(512, kernel_size=3),
            layers.ReLU(),
            layers.Dropout(.5),
            layers.MaxPooling1D(),

            layers.Flatten(),

            layers.Dense(512),
            layers.ReLU(),
            layers.Dropout(.5),

            layers.Dense(256),
            layers.ReLU(),
            layers.Dropout(.5),

            layers.Dense(self.num_of_classes, activation='softmax')
        ])


class LstmModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file = os.path.join(os.path.dirname(__file__), '.lstm_model')

    def create_model(self):
        self.model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.num_of_frames, self.frame_size)),
            layers.LSTM(256, recurrent_dropout=.5, dropout=.5, return_sequences=True),
            layers.LSTM(256, recurrent_dropout=.5, dropout=.5, return_sequences=True),

            #  TemporalMaxPooling(),
            layers.GlobalAveragePooling1D(),

            layers.Dense(128),
            layers.PReLU(),
            layers.Dropout(.5),

            layers.Dense(self.num_of_classes, activation='softmax')
        ])


class TransModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file = os.path.join(os.path.dirname(__file__), '.trans_model')

    def create_model(self):
        inputs = layers.Input(shape=(self.num_of_frames, self.frame_size))

        # Attention and Normalization
        res = inputs
        x = layers.MultiHeadAttention(key_dim=256, num_heads=32,
                                      dropout=.5)(inputs, inputs)
        x = layers.Dropout(.5)(x)
        x = layers.LayerNormalization()(x)
        res += x

        # Feed Forward Part
        x = layers.Conv1D(filters=512, kernel_size=1)(res)
        x = layers.PReLU()(x)
        x = layers.Dropout(.5)(x)

        x = layers.Conv1D(filters=res.shape[-1], kernel_size=1)(x)
        x = layers.Dropout(.5)(x)
        x = layers.LayerNormalization()(x)
        x += res

        x = layers.GlobalAveragePooling1D()(x)

        outputs = layers.Dense(self.num_of_classes, activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)
