#!/usr/bin/env python

import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, callbacks  # type: ignore

from mmwave_gesture.data import DataGenerator, GESTURE


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
            callbacks.EarlyStopping(patience=25, restore_best_weights=True)
        ]
        return model_callbacks

    def compile(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['categorical_accuracy'])

    def train(self, train_paths, y_train, validation_data=None,
                    sample_size=300, batch_size=32, epochs=500):
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

        self.compile()
        history = self.model.fit(train_data(buffer_size=sample_size), epochs=epochs,
                                 steps_per_epoch=len(train_data),
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks=self.get_callbacks())

        history.history['paths'] = {'train': train_paths,
                                    'validation': validation_paths}
        self.history.append(history.history)
        self.save()

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
    def evaluate(self, paths, y, show=False, save=False, batch_size=-1):
        if batch_size < 1:
            batch_size = len(paths)

        y_true, y_pred = [], []
        data = DataGenerator(paths, y, batch_size=batch_size,
                             preprocessor=self.preprocessor)

        for X_batch, y_batch in data():
            y_pred.extend(self._predict(X_batch))
            y_true.extend(y_batch)

        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        print(f'Accuracy: {accuracy_score(y_true_labels, y_pred_labels)}')
        print(f'Balanced accuracy:', end=' ')
        print(balanced_accuracy_score(y_true_labels, y_pred_labels))
        print(f'f1: {f1_score(y_true_labels, y_pred_labels, average="macro")}')
        print(f'Recall: {recall_score(y_true_labels, y_pred_labels, average="macro")}')

        if save or show:
            labels = [GESTURE[i].name for i in np.unique([y_true_labels,
                                                          y_pred_labels])]
            cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')
            ConfusionMatrixDisplay(cm, display_labels=labels).plot()
            plt.title(f'Confusion matrix for {model.__class__.__name__}')

            if show:
                plt.show()

            if save:
                plt.savefig(os.path.join(self.dir, 'confusion_matrix.png'))

        return y_true_labels, y_pred

    @check_model
    def predict(self, sample, preprocess=True):
        if preprocess and self.preprocessor is not None:
            for p in self.preprocessor:
                sample = p.process(sample)

        return self._predict(np.array([sample]))[0]

    @tf.function
    def _predict(self, X):
        return self.model(X, training=False)


class ConvModel(Model):
    def get_regularizer(self, value):
        if value > 0:
            return tf.keras.regularizers.l2(value)
        return None

    def get_conv(self, conv_type):
        if conv_type.lower() == '1d':
            return layers.Conv1D, layers.MaxPooling1D
        elif conv_type.lower() == '2d':
            return layers.Conv2D, layers.MaxPooling2D
        else:
            raise('Unsupported conv type.')

    def conv_block(self, conv_type, depth=1, start_filters=128, kernel_size=3,
                         factor=2, factor_step=1, downsample=2,
                         downsample_step=1, l2=-1):
        def model(x):
            conv_layer, max_pooling_layer = self.get_conv(conv_type)

            for i in range(depth):
                filters = start_filters * factor**(i//factor_step)
                x = conv_layer(filters, kernel_size=kernel_size, padding='same',
                               kernel_regularizer=self.get_regularizer(l2))(x)
                x = layers.ReLU()(x)

                do_downsample = downsample_step == 1 or (i+1) % downsample_step == 0
                if downsample > 1 and do_downsample:
                    x = max_pooling_layer(downsample)(x)

            return x
        return model

    def res_block(self, conv_type, filters=128, depth=3, kernel_size=3, l2=-1):
        def model(x):
            conv_layer, _ = self.get_conv(conv_type)

            for _ in range(depth):
                strides = filters//x.shape[-1]
                if filters % x.shape[-1] != 0:
                    raise ValueError('Filter size error')

                res = x
                x = conv_layer(filters, kernel_size=kernel_size,
                               strides=strides, padding='same',
                               kernel_regularizer=self.get_regularizer(l2))(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)

                x = conv_layer(filters, kernel_size=kernel_size, padding='same',
                               kernel_regularizer=self.get_regularizer(l2))(x)
                if strides > 1:
                    res = conv_layer(filters=filters, kernel_size=1, strides=strides,
                                     kernel_regularizer=self.get_regularizer(l2))(res)

                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
                x += res

            return x
        return model


class Conv1DModel(ConvModel):
    def create_model(self, depth=3, l2=-1, units=-1, dropout=.5, **kwargs):
        input = layers.Input(self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)

        x = self.conv_block('1d', depth=depth, l2=l2, **kwargs)(x)
        x = layers.Flatten()(x)

        if units > 0:
            x = layers.Dense(units)(x)
            x = layers.ReLU()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

        self.model = tf.keras.Model(input, self.output(x))


class Conv2DModel(ConvModel):
    def create_model(self, depth=3, l2=-1, units=-1, dropout=.5, **kwargs):
        input = layers.Input(self.X_shape)

        x = self.conv_block('2d', depth=depth, l2=l2, **kwargs)(input)
        x = layers.Flatten()(x)

        if units > 0:
            x = layers.Dense(units)(x)
            x = layers.ReLU()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

        self.model = tf.keras.Model(input, self.output(x))


class ResNet1DModel(ConvModel):
    def create_model(self, blocks=[2, 2, 2, 2], start_filters=128, factor=2, l2=-1):
        input = layers.Input(self.X_shape)
        x = layers.Reshape((self.X_shape[0], -1))(input)
        x = self.conv_block('1d', start_filters=start_filters, kernel_size=5, l2=l2)(x)

        for i, block in enumerate(blocks):
            x = self.res_block('1d', filters=start_filters * factor**i,
                               depth=block, l2=l2)(x)

        x = layers.Flatten()(x)
        self.model = tf.keras.Model(input, self.output(x))


class ResNet2DModel(ConvModel):
    def create_model(self, blocks=[2, 2, 2, 2], start_filters=128, factor=2, l2=-1):
        input = layers.Input(self.X_shape)
        x = self.conv_block('2d', start_filters=start_filters,
                            kernel_size=5, l2=l2)(input)

        for i, block in enumerate(blocks):
            x = self.res_block('2d', filters=start_filters * factor**i,
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
        i, d_model = tf.cast(i, tf.float32), tf.cast(d_model, tf.float32)
        return pos/tf.pow(10000., (2*(i//2))/d_model)

    def call(self, input):
        batch_size = tf.shape(input)[0]
        num_of_frames = tf.shape(input)[1]
        num_of_objs = tf.shape(input)[2]
        d_model = tf.shape(input)[3]

        angles = self.get_angles(
            tf.range(num_of_frames, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model*2  # odd d_model fix
        )

        encoding = tf.concat([tf.math.sin(angles[:, 0::2]),
                              tf.math.cos(angles[:, 1::2])], axis=1)
        encoding = encoding[:, :d_model]  # odd d_model fix

        encoding = tf.reshape(encoding, shape=(angles.shape[0], -1))

        encoding = tf.tile(encoding[tf.newaxis, :, tf.newaxis],
                           [batch_size, 1, num_of_objs, 1])

        return input + encoding


class RelativePositionalEncoding(layers.Layer):
    def __init__(self):
        super(RelativePositionalEncoding, self).__init__()
        self.pos_emb = None

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            'pos_emb',
            shape=[1, input_shape[1], input_shape[2], input_shape[3]],
            initializer=tf.initializers.RandomNormal(),
            trainable=True,
        )

    def call(self, input):
        if self.pos_emb is None:
            self.build(input.shape)

        batch_size = tf.shape(input)[0]
        return input + self.pos_emb*tf.ones([batch_size, 1, 1, 1], dtype=tf.float32)


class UnorderedObjectsEncoding(layers.Layer):
    def __init__(self):
        super(UnorderedObjectsEncoding, self).__init__()
        self.obj_emb = None

    def build(self, input_shape):
        self.obj_emb = self.add_weight(
            'obj_emb',
            shape=[input_shape[2], input_shape[3]],
            initializer=tf.initializers.RandomNormal(),
            trainable=True
        )

    def call(self, input):
        if self.obj_emb is None:
            self.build(input.shape)

        return input + self.obj_emb


class TransModel(Model):
    def create_model(self, key_dim=64, num_heads=8, depth=2, dropout=.5):
        input = layers.Input(shape=self.X_shape)
        x = RelativePositionalEncoding()(input)
        x = UnorderedObjectsEncoding()(x)
        x = layers.Reshape((self.X_shape[0], -1))(x)

        # Attention and Normalization
        for _ in range(depth):
            res = x
            x = layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads,
                                          dropout=dropout)(x, x)
            x += res
            x = layers.LayerNormalization()(x)

        x = layers.Flatten()(x)
        self.model = tf.keras.Model(input, self.output(x))

    def load(self):
        custom_objects = {'RelativePositionalEncoding': RelativePositionalEncoding,
                          'UnorderedObjectsEncoding': UnorderedObjectsEncoding}
        super().load(custom_objects=custom_objects)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    from mmwave_gesture.data import Logger, Formats, DataGenerator
    from mmwave_gesture.data.preprocessor import Polar, ZeroPadd


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

    preprocessor = [Polar(Formats(config)),
                    ZeroPadd(num_of_frames=25, num_of_objs=40)]

    for GestureModel in [Conv1DModel, Conv2DModel, LstmModel,
                         TransModel, ResNet1DModel, ResNet2DModel]:
        model = GestureModel(preprocessor=preprocessor)

        model.train(train_paths, y_train, validation_data=(val_paths, y_val))
        model.load()
        model.evaluate(test_paths, y_test, save=True)
        del model
