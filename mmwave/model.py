#!/usr/bin/env python

import os
import pickle
from abc import ABC, abstractmethod

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, callbacks  # type: ignore

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Model(ABC):
    def __init__(self, preprocessor=None, **params):
        self.params = params
        self.preprocessor = preprocessor
        if preprocessor is not None and not isinstance(preprocessor, list):
            self.preprocessor = [preprocessor]

        self.history = []
        self.model = None

        self.dir = os.path.join(os.path.dirname(__file__),
                                f'.{self.__class__.__name__}')
        os.makedirs(self.dir, exist_ok=True)

        self.X_shape = None
        self.y_shape = None

    @abstractmethod
    def create_model(self):
        pass

    def output(self, x):
        return layers.Dense(self.y_shape, activation='softmax')(x)

    def get_callbacks(self):
        model_callbacks = [
            callbacks.ReduceLROnPlateau(factor=.5, patience=5, verbose=1),
            callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        ]
        return model_callbacks

    def train(self, train_paths, y_train, validation_data=None, batch_size=32):
        train_data = DataGenerator(train_paths, y_train, batch_size=batch_size,
                                   preprocessor=self.preprocessor,
                                   repeat=True, shuffle=True)

        self.X_shape, self.y_shape = train_data.X_shape, train_data.y_shape[-1]

        validation_paths, validation_steps = None, None
        if validation_data is not None:
            validation_paths = validation_data[0]
            validation_data = DataGenerator(*validation_data, repeat=True,
                                            preprocessor=self.preprocessor)
            validation_steps = len(validation_data)
            validation_data = validation_data()

        tf.keras.backend.clear_session()
        self.create_model(**self.params)
        self.model.summary()
        tf.keras.utils.plot_model(self.model, show_shapes=True, expand_nested=True,
                                  to_file=os.path.join(self.dir, 'model.png'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['categorical_accuracy'])

        history = self.model.fit(train_data(buffer_size=300), epochs=500,
                                 steps_per_epoch=len(train_data),
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks=self.get_callbacks())

        history.history['paths'] = {'train': train_paths,
                                    'validation': validation_paths}
        self.history.append(history.history)
        self.save()

        return history

    def load(self):
        print('Loading model...', end='')
        self.model = tf.keras.models.load_model(os.path.join(self.dir, 'model.h5'))

        with open(os.path.join(self.dir, 'preprocessor'), 'rb') as f:
            self.preprocessor = pickle.load(f)

        with open(os.path.join(self.dir, 'history'), 'rb') as f:
            self.history = pickle.load(f)

        # Initilze model with zero input
        self.predict(np.zeros(self.model.input_shape[1:]), preprocess=False)
        print(f'{Fore.GREEN}Done.')

    def save(self):
        print(f'{Fore.YELLOW}Saving model...', end='', flush=True)
        self.model.save(os.path.join(self.dir, 'model.h5'))

        with open(os.path.join(self.dir, 'preprocessor'), 'wb') as f:
            pickle.dump(self.preprocessor, f)

        with open(os.path.join(self.dir, 'history'), 'wb') as f:
            pickle.dump(self.history, f)

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
        if self.preprocessor is not None:
            data = [self.preprocessor.process(d) for d in data]

        preds = self.model.evaluate([self.preprocessor.process(d) for d in data])
        print(f'Loss: {round(preds[0], 4)}', end=' ')
        print(f'Acc: {round(preds[1], 4)}')

    @check_model
    def predict(self, X, preprocess=True):
        if preprocess and self.preprocessor is not None:
            for p in self.preprocessor:
                X = p.process(X)

        return self._predict(np.array([X]))[0]

    @tf.function
    def _predict(self, X):
        return self.model(X, training=False)


class ConvModel(Model):
    def create_model(self, start_filters=128, factor=2, factor_step=1,
                           kernel_size=3, depth=3, units=64, dropout=.5):
        input = layers.Input(self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        for i in range(depth):
            x = layers.Conv1D(start_filters * factor**(i//factor_step),
                              kernel_size=kernel_size, padding='same')(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(dropout)(x)
            x = layers.MaxPooling1D()(x)

        x = layers.Flatten()(x)

        x = layers.Dense(units)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)

        self.model = tf.keras.Model(input, self.output(x))


class LstmModel(Model):
    def create_model(self, units=128, depth=2, dense_units=64, dropout=.5):
        input = layers.Input(shape=self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        for i in range(depth):
            return_sequences = True if i != depth-1 else False
            x = layers.LSTM(units, return_sequences=return_sequences,
                            recurrent_dropout=dropout, dropout=dropout)(x)

        if dense_units > 0:
            x = layers.Dense(dense_units)(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(dropout)(x)

        self.model = tf.keras.Model(input, self.output(x))


class PositionalEncoding(layers.Layer):
    def get_angles(self, pos, i, d_model):
        i, d_model = tf.cast(i, tf.float32), tf.cast(d_model, tf.float32)
        return pos/tf.pow(10000.0, (2*(i//2))/d_model)

    def call(self, input):
        batch_size = tf.shape(input)[0]
        num_of_frames = tf.shape(input)[1]
        num_of_objs = tf.shape(input)[2]
        d_model = tf.shape(input)[3]

        angles = self.get_angles(
            tf.range(num_of_frames, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )

        encoding = tf.stack([tf.math.sin(angles[:, 0::2]),
                             tf.math.cos(angles[:, 1::2])], axis=-1)
        encoding = tf.reshape(encoding, shape=(angles.shape[0], -1))

        encoding = tf.tile(encoding[tf.newaxis, :, tf.newaxis],
                           [batch_size, 1, num_of_objs, 1])

        return input + encoding


class TransModel(Model):
    def create_model(self, key_dim=128, num_heads=4, depth=1, filters=64, dropout=.5):
        input = layers.Input(shape=self.X_shape)
        x = PositionalEncoding()(input)
        x = layers.Reshape((self.X_shape[0], -1))(x)

        # Attention and Normalization
        for _ in range(depth):
            res = x
            x = layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads,
                                          dropout=dropout)(x, x)
            x += res
            x = layers.LayerNormalization()(x)

        # Feed-forward
        if filters > 0:
            res = x
            x = layers.Conv1D(filters=filters, kernel_size=3, padding='same')(x)
            x = layers.ReLU()(x)

            x = layers.Conv1D(filters=res.shape[-1], kernel_size=1)(x)
            x += res
            x = layers.LayerNormalization()(x)

        x = layers.Flatten()(x)
        self.model = tf.keras.Model(input, self.output(x))


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    from mmwave.data import Logger, Formats, DataGenerator
    from mmwave.data.preprocessor import Polar, ZeroPadd

    paths, y = Logger.get_paths()
    train_paths, test_paths, y_train, y_test = train_test_split(paths, y, stratify=y,
                                                                test_size=.3,
                                                                random_state=12)

    val_paths, test_paths, y_val, y_test = train_test_split(test_paths, y_test,
                                                            stratify=y_test,
                                                            test_size=.5,
                                                            random_state=12)

    config = os.path.join(os.path.dirname(__file__),
                          'communication/profiles/profile.cfg')
    preprocessor = [Polar(Formats(config)), ZeroPadd()]

    # model = ConvModel(preprocessor=preprocessor)
    # model = LstmModel(preprocessor=preprocessor)
    model = TransModel(preprocessor=preprocessor)
    model.train(train_paths, y_train, validation_data=(val_paths, y_val))
