#!/usr/bin/env python

import os
from abc import ABC, abstractmethod

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, callbacks  # type: ignore

from mmwave.data import GESTURE
from mmwave.data import PolarPreprocessor

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Model(ABC):
    def __init__(self, preprocessor, **params):
        self.params = params
        self.preprocessor = preprocessor
        if preprocessor is not None and not isinstance(preprocessor, list):
            self.preprocessor = [preprocessor]

        self.model = None
        self.dir = os.path.join(os.path.dirname(__file__),
                                f'.{self.__class__.__name__}')
        self.X_shape = None
        self.y_shape = None

    @abstractmethod
    def create_model(self):
        pass

    def get_callbacks(self):
        model_callbacks = [
            callbacks.ReduceLROnPlateau(factor=.5, patience=10, verbose=1),
            callbacks.EarlyStopping(patience=30, restore_best_weights=True)
        ]
        return model_callbacks

    def train(self, train_paths, y_train, validation_data=None):
        train_data = DataGenerator(train_paths, y_train,
                                   preprocessor=self.preprocessor,
                                   repeat=True, shuffle=True)

        self.X_shape, self.y_shape = train_data.X_shape, train_data.y_shape[-1]

        validation_steps = None
        if validation_data is not None:
            validation_data = DataGenerator(*validation_data,
                                            preprocessor=self.preprocessor)
            validation_steps = len(validation_data)
            validation_data = validation_data()

        tf.keras.backend.clear_session()
        self.create_model(**self.params)
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['categorical_accuracy'])

        self.model.fit(train_data(buffer_size=300), epochs=1000,
                       steps_per_epoch=len(train_data),
                       validation_data=validation_data,
                       validation_steps=validation_steps,
                       callbacks=self.get_callbacks())

    def load(self):
        print('Loading model...', end='')
        self.model = tf.keras.models.load_model(self.dir)
        # TODO: pass random input through model to initilize it
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
    def evaluate(self, data):
        preds = self.model.evaluate(data)
        print(f'Loss: {round(preds[0], 4)}', end=' ')
        print(f'Acc: {round(preds[1], 4)}')

    @check_model
    def predict(self, X, preprocess=True):
        if preprocess and self.preprocessor is not None:
            for p in self.preprocessor:
                X = p.process(X)

        y_pred = self._predict(X)

        best_guess = GESTURE[np.argmax(y_pred)]
        # best_value = sorted(y_pred[0], reverse=True)

        return best_guess

    @tf.function
    def _predict(self, X):
        return self.model(X, training=False)


class ConvModel(Model):
    def create_model(self, start_filters=128, factor=2, kernel_size=3, depth=3,
                           units=256, dropout=.5):
        input = layers.Input(self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        conv_size = start_filters
        x = layers.Conv1D(conv_size, kernel_size=kernel_size)(x)
        x = layers.Conv1D(conv_size, kernel_size=kernel_size)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D()(x)

        for _ in range(depth-1):
            conv_size *= factor
            x = layers.Conv1D(conv_size, kernel_size=kernel_size)(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(dropout)(x)
            x = layers.MaxPooling1D()(x)

        x = layers.Flatten()(x)

        x = layers.Dense(units)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)

        outout = layers.Dense(self.y_shape, activation='softmax')(x)
        self.model = tf.keras.Model(input, outout)


class LstmModel(Model):
    def create_model(self, units=128, depth=2, dense_units=64, dropout=.5):
        input = layers.Input(shape=self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        for i in range(depth):
            return_sequences = True if i != depth-1 else False
            x = layers.LSTM(units, return_sequences=return_sequences,
                            dropout=dropout, recurrent_dropout=dropout)(x)

        x = layers.Dense(dense_units)(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)

        output = layers.Dense(self.y_shape, activation='softmax')(x)
        self.model = tf.keras.Model(input, output)


class TransModel(Model):
    def create_model(self, key_dim=64, num_heads=4):
        input = layers.Input(shape=self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        # Attention and Normalization
        res = x
        x = layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads,
                                      dropout=.5)(x, x)
        x = layers.Dropout(.5)(x)
        x = layers.LayerNormalization()(x)
        res += x

        # Feed Forward Part
        x = layers.Conv1D(64, kernel_size=1)(res)
        x = layers.PReLU()(x)
        x = layers.Dropout(.5)(x)

        x = layers.Conv1D(filters=res.shape[-1], kernel_size=1)(x)
        x = layers.Dropout(.5)(x)
        x = layers.LayerNormalization()(x)
        x += res

        x = layers.Flatten()(x)

        output = layers.Dense(self.y_shape, activation='softmax')(x)
        self.model = tf.keras.Model(input, output)


class TestModel(Model):
    def create_model(self):
        input = layers.Input(shape=self.X_shape)

        x = layers.TimeDistributed(layers.Conv1D(32, 65))(input)

        # x = layers.TimeDistributed(layers.Conv1D(32, 16))(input)
        # x = layers.ReLU()(x)

        # x = layers.TimeDistributed(layers.Conv1D(32, 16))(x)
        # x = layers.ReLU()(x)

        # x = layers.TimeDistributed(layers.Conv1D(64, 16))(x)
        # x = layers.ReLU()(x)

        # x = layers.TimeDistributed(layers.Conv1D(64, 16))(x)
        # x = layers.ReLU()(x)

        # x = layers.TimeDistributed(layers.Conv1D(128, 5))(x)
        # x = layers.ReLU()(x)

        x = layers.Reshape((self.X_shape[0], -1))(x)

        x = layers.LSTM(64, return_sequences=True,
                        dropout=0.2, recurrent_dropout=0.2)(x)
        x = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)

        output = layers.Dense(self.y_shape, activation='softmax')(x)
        self.model = tf.keras.Model(input, output)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    from mmwave.data import Logger, Formats, DataGenerator, PolarPreprocessor


    paths, y = Logger.get_paths()
    paths = paths*500
    y = y*500

    train_paths, test_paths, y_train, y_test = train_test_split(paths, y, stratify=y,
                                                                test_size=0.3)

    val_paths, test_paths, y_val, y_test = train_test_split(test_paths, y_test,
                                                            stratify=y_test,
                                                            test_size=0.5)

    formats = Formats('../mmwave/communication/profiles/profile.cfg')
    preprocessor = PolarPreprocessor(formats)

    model = ConvModel(preprocessor=preprocessor)
    # model = LstmModel(preprocessor=preprocessor)
    # model = TransModel(preprocessor=preprocessor)
    model.train(train_paths, y_train, validation_data=(val_paths, y_val))
