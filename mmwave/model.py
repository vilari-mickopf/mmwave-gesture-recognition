#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from abc import ABC, abstractmethod

import numpy as np

import tensorflow as tf
from tensorflow.keras import utils, preprocessing, layers, callbacks  # type: ignore

from mmwave.data.formats import GESTURE

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Model(ABC):
    def __init__(self, num_of_frames=50, num_of_objs=65, num_of_data_in_obj=5, gesture=GESTURE):
        self.model = None
        self.model_file = os.path.join(os.path.dirname(__file__),
                                       f'.{self.__class__.__name__}')

        self.num_of_frames = num_of_frames
        self.num_of_objs = num_of_objs
        self.num_of_data_in_obj = num_of_data_in_obj

        self.gesture = gesture
        self.num_of_classes = len(gesture)

        self.frame_size = self.num_of_objs*self.num_of_data_in_obj

    def padd_data(self, data):
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

    def prep_data(self, X, y=None):
        # X has been already normalized while importing the data from .csv files
        X = self.padd_data(X)
        X = X.reshape((len(X), self.num_of_frames, self.frame_size))

        if y is not None:
            y = utils.to_categorical(y)
            return X, y

        return X

    @abstractmethod
    def create_model(self):
        pass

    def train(self, X_train, y_train, X_val, y_val):
        X_train, y_train = self.prep_data(X_train, y_train)
        X_val, y_val = self.prep_data(X_val, y_val)

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
        print('Loading model...', end='')
        self.model = tf.keras.models.load_model(self.model_file)
        print(f'{Fore.GREEN}Done.')

    def loaded(self):
        return self.model is not None

    def check_model(func):
        def wrapper(self, *args, **kwargs):
            if self.model is None:
                print(f'{Fore.RED}Model not created.')
                return
            return func(self, *args, **kwargs)
        return wrapper

    @check_model
    def evaluate(self, X, y):
        X, y = self.prep_data(X, y)
        preds = self.model.evaluate(X, y)
        print(f'Loss: {round(preds[0], 4)}', end=' ')
        print(f'Acc: {round(preds[1], 4)}')

    @check_model
    def predict(self, X, debug=False):
        y_pred = self.model.predict(self.prep_data(X))
        best_guess = [y_pred[0].tolist().index(x) for x in sorted(y_pred[0], reverse=True)]
        best_value = sorted(y_pred[0], reverse=True)

        if debug:
            for guess, val in zip(best_guess, best_value):
                print(f'{Fore.YELLOW}Best guess: {self.gesture(guess).name.lower()}: {val:.2f}')
            print(f'{Fore.CYAN}------------------------------\n')

        if best_value[0] >= .9:
            print(f'{Fore.GREEN}Gesture recognized:',
                  f'{Fore.BLUE}{self.gesture(best_guess[0]).name.lower()}')
            print(f'{Fore.CYAN}==============================\n')


class ConvModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def create_model(self):
        self.model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.num_of_frames, self.frame_size)),
            layers.LSTM(256, recurrent_dropout=.5, dropout=.5, return_sequences=True),
            layers.LSTM(256, recurrent_dropout=.5, dropout=.5, return_sequences=True),

            layers.GlobalAveragePooling1D(),

            layers.Dense(128),
            layers.PReLU(),
            layers.Dropout(.5),

            layers.Dense(self.num_of_classes, activation='softmax')
        ])


class TransModel(Model):
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
