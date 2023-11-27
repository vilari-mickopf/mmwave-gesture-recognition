#!/usr/bin/env python

import os
import pickle
import inspect
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, callbacks  # type: ignore

from mmwave_gesture.data import DataGenerator, GESTURE, Logger, Formats
from mmwave_gesture.data.preprocessor import Polar, ConsistentShape

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Model(ABC):
    def __init__(self, preprocessor=None, **params):
        self.preprocessor = preprocessor
        if preprocessor is not None and not isinstance(preprocessor, list):
            self.preprocessor = [preprocessor]

        self.params = params

        self.history = []
        self.model = None
        self.X_shape = None
        self.y_shape = None

        self.set_dir()

    def get_params(self):
        bound_args = inspect.signature(self.create_model).bind(**self.params)
        bound_args.apply_defaults()
        params = bound_args.arguments
        if params.get('self') is not None:
            params.pop('self')

        return params

    def set_dir(self):
        self.dir = os.path.join(os.path.dirname(__file__), 'models',
                                f'{self.__class__.__name__}')
        os.makedirs(self.dir, exist_ok=True)

    @abstractmethod
    def create_model(self):
        pass

    def output(self, x):
        activation = 'softmax' if self.y_shape > 1 else 'sigmoid'
        return layers.Dense(self.y_shape, activation=activation)(x)

    def get_callbacks(self, verbose):
        model_callbacks = [
            callbacks.ReduceLROnPlateau(factor=3/4, patience=5, verbose=verbose),
            callbacks.EarlyStopping(patience=25, restore_best_weights=True)
        ]
        return model_callbacks

    def compile(self):
        loss = 'categorical_crossentropy' if self.y_shape > 1 else 'binary_crossentropy'
        self.model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    def init_data_gen(self, train_data, validation_data, batch_size):
        self.train_data = DataGenerator(*train_data, preprocessor=self.preprocessor,
                                        batch_size=batch_size, repeat=True, shuffle=True)

        self.X_shape, self.y_shape = self.train_data.X_shape, self.train_data.y_shape[-1]

        self.validation_data = DataGenerator(*validation_data, preprocessor=self.preprocessor)

    def train(self, train_data, validation_data, batch_size=32, epochs=500,
                    save=True, verbose=True):
        self.init_data_gen(train_data, validation_data, batch_size)

        tf.keras.backend.clear_session()
        self.create_model(**self.params)
        if verbose:
            self.model.summary()
        self.compile()

        history = self.model.fit(self.train_data(), epochs=epochs,
                                 steps_per_epoch=len(self.train_data),
                                 validation_data=self.validation_data(),
                                 validation_steps=len(self.validation_data),
                                 callbacks=self.get_callbacks(verbose),
                                 verbose=verbose)

        history.history['paths'] = {'train': train_data[0],
                                    'validation': validation_data[0]}
        self.history.append(history.history)

        if save:
            self.save()
            tf.keras.utils.plot_model(self.model, show_shapes=True, expand_nested=True,
                                      to_file=os.path.join(self.dir, 'model.png'))

        return history

    def load(self, component=None, custom_objects=None):
        if component is not None:
            return self.load_component(component)

        self.load_model(custom_objects)

    def load_model(self, custom_objects):
        tf.keras.backend.clear_session()
        print('Loading model...', end='')
        self.model = tf.keras.models.load_model(os.path.join(self.dir, 'model.h5'),
                                                custom_objects=custom_objects)

        self.load_component('preprocessor')
        self.load_component('history')

        # Pass zero input to initilze the model
        self.predict(np.zeros(self.model.input_shape[1:]), preprocess=False)
        print(f'{Fore.GREEN}Done.', flush=True)

    def load_component(self, component):
        path = os.path.join(self.dir, component)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                setattr(self, component, pickle.load(f))

    def save(self):
        print(f'{Fore.YELLOW}Saving model...', end='', flush=True)
        self.model.save(os.path.join(self.dir, 'model.h5'))

        self.save_component('preprocessor')
        self.save_component('history')

        print(f'{Fore.GREEN}Done.')

    def save_component(self, component):
        value = getattr(self, component)
        if value is None:
            return

        with open(os.path.join(self.dir, component), 'wb') as f:
            pickle.dump(value, f)

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
    def evaluate(self, test_data, verbose=True, show=False, save=False):
        y_true, y_pred = [], []
        data = DataGenerator(*test_data, preprocessor=self.preprocessor)

        for X_batch, y_batch in data():
            pred = self._predict(X_batch)
            if pred is None:
                continue

            y_pred.extend(pred)
            y_true.extend(y_batch)

        if len(y_pred) < 1:
            return

        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        if verbose:
            print(f'Accuracy: {accuracy_score(y_true_labels, y_pred_labels)}')
            print(f'Balanced accuracy:', end=' ')
            print(balanced_accuracy_score(y_true_labels, y_pred_labels))
            print(f'f1: {f1_score(y_true_labels, y_pred_labels, average="macro")}')
            print(f'Recall: {recall_score(y_true_labels, y_pred_labels, average="macro")}')

        if save or show:
            labels = [GESTURE[i].name for i in np.unique([y_true_labels, y_pred_labels])]
            cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')
            ConfusionMatrixDisplay(cm, display_labels=labels).plot()
            plt.title(f'Confusion matrix for {self.__class__.__name__}')

            if show:
                plt.show()

            if save:
                plt.savefig(os.path.join(self.dir, 'confusion_matrix.png'))

        return np.array(y_true_labels), np.array(y_pred)

    @check_model
    def predict(self, sample, preprocess=True):
        if preprocess:
            sample = DataGenerator.preprocess(sample, self.preprocessor)
            if sample is None:
                return None

        return self._predict(np.array([sample]))[0]

    @tf.function
    def _predict(self, X):
        return self.model(X, training=False)


class ConvModel(Model):
    LAYERS = {
        '1d': {'conv': layers.Conv1D,
               'maxpool': layers.MaxPooling1D},
        '2d': {'conv': layers.Conv2D,
               'maxpool': layers.MaxPooling2D},
    }

    def get_regularizer(self, value):
        if value > 0:
            return tf.keras.regularizers.l2(value)
        return None

    def conv_block(self, conv_type, depth=1, start_filters=128, kernel_size=3,
                         factor=2, factor_step=1, downsample=2, downsample_step=1, l2=-1):
        def model(x):
            conv = self.LAYERS[conv_type]['conv']
            maxpool = self.LAYERS[conv_type]['maxpool']
            for i in range(depth):
                filters = start_filters * factor**(i//factor_step)
                x = conv(filters, kernel_size=kernel_size, padding='same',
                         kernel_regularizer=self.get_regularizer(l2))(x)
                x = layers.ReLU()(x)

                do_downsample = downsample_step == 1 or (i+1) % downsample_step == 0
                if downsample > 1 and do_downsample and i != depth-1:
                    x = maxpool(downsample)(x)

            return x
        return model

    def res_block(self, conv_type, filters=128, depth=3, kernel_size=3, l2=-1):
        def model(x):
            conv = self.LAYERS[conv_type]['conv']

            for _ in range(depth):
                strides = filters//x.shape[-1]
                if filters % x.shape[-1] != 0:
                    raise ValueError('Filter size error')

                res = x
                x = conv(filters, kernel_size=kernel_size,
                         strides=strides, padding='same',
                         kernel_regularizer=self.get_regularizer(l2))(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)

                x = conv(filters, kernel_size=kernel_size, padding='same',
                         kernel_regularizer=self.get_regularizer(l2))(x)
                if strides > 1:
                    res = conv(filters=filters, kernel_size=1, strides=strides,
                               kernel_regularizer=self.get_regularizer(l2))(res)

                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
                x += res

            return x
        return model


class Conv1DModel(ConvModel):
    def create_model(self, start_filters=64, kernel_size=3, factor=2, factor_step=1,
                           downsample=2, downsample_step=1, depth=3, units=-1,
                           l2=-1,  dropout=.5):
        input = layers.Input(self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        x = self.conv_block('1d', start_filters=start_filters, kernel_size=kernel_size,
                            factor=factor, factor_step=factor_step,
                            downsample=downsample, downsample_step=downsample_step,
                            depth=depth, l2=l2)(x)
        x = layers.Flatten()(x)

        if units > 0:
            x = layers.Dense(units)(x)
            x = layers.ReLU()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

        self.model = tf.keras.Model(input, self.output(x))


class Conv2DModel(ConvModel):
    def create_model(self, start_filters=64, kernel_size=3, factor=2, factor_step=1,
                           downsample=2, downsample_step=1, depth=3, units=-1,
                           l2=-1,  dropout=.5):
        input = layers.Input(self.X_shape)

        x = self.conv_block('2d', start_filters=start_filters, kernel_size=kernel_size,
                            factor=factor, factor_step=factor_step,
                            downsample=downsample, downsample_step=downsample_step,
                            depth=depth, l2=l2)(input)
        x = layers.Flatten()(x)

        if units > 0:
            x = layers.Dense(units)(x)
            x = layers.ReLU()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

        self.model = tf.keras.Model(input, self.output(x))


class ResNet1DModel(ConvModel):
    def create_model(self, blocks=[2, 2, 2, 2], start_filters=128, l2=-1):
        input = layers.Input(self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)
        x = self.conv_block('1d', start_filters=start_filters, kernel_size=5, l2=l2)(x)

        for i, block in enumerate(blocks):
            x = self.res_block('1d', filters=start_filters * 2**i,
                               depth=block, l2=l2)(x)

        x = layers.Flatten()(x)
        self.model = tf.keras.Model(input, self.output(x))


class ResNet2DModel(ConvModel):
    def create_model(self, blocks=[2, 2, 2, 2], start_filters=64, l2=-1):
        input = layers.Input(self.X_shape)
        x = self.conv_block('2d', start_filters=start_filters,
                            kernel_size=5, l2=l2)(input)

        for i, block in enumerate(blocks):
            x = self.res_block('2d', filters=start_filters * 2**i,
                               depth=block, l2=l2)(x)

        x = layers.Flatten()(x)
        self.model = tf.keras.Model(input, self.output(x))


class LstmModel(Model):
    def create_model(self, units=256, depth=2, dense_units=-1, dropout=.5):
        input = layers.Input(shape=self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        for _ in range(depth):
            x = layers.LSTM(units, return_sequences=True,
                            recurrent_dropout=dropout, dropout=dropout)(x)

        x = layers.Flatten()(x)

        if dense_units > 0:
            x = layers.Dense(dense_units)(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(dropout)(x)

        self.model = tf.keras.Model(input, self.output(x))


class PositionalEncoding(layers.Layer):
    def get_angles(self, pos, i, d_model):
        d_model = tf.cast(d_model, tf.float32)
        return pos/tf.pow(10000.0, (2*(i//2))/d_model)

    def call(self, input):
        num_of_frames = tf.shape(input)[1]
        d_model = tf.shape(input)[2]

        angles = self.get_angles(
            tf.range(num_of_frames, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )

        encoding = tf.concat([tf.math.sin(angles[:, 0::2]),
                              tf.math.cos(angles[:, 1::2])], axis=-1)
        encoding = tf.reshape(encoding, (num_of_frames, -1))
        encoding = encoding[tf.newaxis, :, :]

        return input + encoding[:, :num_of_frames, :]


class TransModel(Model):
    def create_model(self, key_dim=16, num_heads=8, d_model=128, depth=2, dropout=.5):
        input = layers.Input(shape=self.X_shape)

        # Encondig
        x = layers.TimeDistributed(layers.Flatten())(input)
        x = layers.Dense(d_model)(x)
        x = PositionalEncoding()(x)

        # Attention and Normalization
        for _ in range(depth):
            res = x
            x = layers.LayerNormalization()(x)
            x = layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads,
                                          dropout=dropout)(x, x)
            x = layers.Dropout(dropout)(x)
            x += res
            x = layers.LayerNormalization()(x)

            res = x
            x = layers.Dense(d_model)(x)
            x = layers.ReLU()(x)
            x = layers.Dense(d_model)(x)
            x = layers.Dropout(dropout)(x)
            x += res

        x = layers.Flatten()(x)
        self.model = tf.keras.Model(input, self.output(x))

    def load(self):
        custom_objects = {'PositionalEncoding': PositionalEncoding}
        super().load(custom_objects=custom_objects)


def get_model_types(base=Model):
    subclasses = {}
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, base) and obj is not base:
            # Check if not abstract
            if not hasattr(obj, '__abstractmethods__') or len(obj.__abstractmethods__) == 0:
                subclasses[name.lower().replace('model', '')] = obj
    return subclasses


def train_evaluate_model(model, dir=None):
    paths, y = Logger.get_paths(dir)
    train_data, val_data, test_data = Logger.split(paths, y)

    model.train(train_data, validation_data=val_data)
    model.load()
    model.evaluate(test_data, save=True)


if __name__ == '__main__':
    config = os.path.join(os.path.dirname(__file__),
                          'communication/profiles/profile.cfg')

    preprocessor = [Polar(Formats(config)),
                    ConsistentShape(num_of_frames=25, num_of_objs=40)]

    for GestureModel in get_model_types().values():
        model = GestureModel(preprocessor=preprocessor)
        train_evaluate_model(model)
        del model
