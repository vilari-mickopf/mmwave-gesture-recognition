#!/usr/bin/env python

from copy import deepcopy

import numpy as np

import sklearn
import tensorflow as tf

from mmwave.data import GESTURE


class DataGenerator:
    def __init__(self, paths, y, preprocessor=None, batch_size=32,
                       shuffle=False, repeat=False):
        self.paths = paths
        self.y = y

        self.preprocessor = preprocessor
        if preprocessor is not None and not isinstance(preprocessor, list):
            self.preprocessor = [preprocessor]

        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle

        self.X_shape = self.load(self.paths[0]).shape
        self.y_shape = self.get_target(y[0]).shape

    def load(self, path, preprocess=True):
        data = np.load(path, allow_pickle=True)['data']
        if preprocess and self.preprocessor is not None:
            for p in self.preprocessor:
                data = p.process(data)

        return data

    def get_target(self, label):
        return tf.keras.utils.to_categorical(label, num_classes=len(GESTURE))

    def get_data(self):
        file_index = 0
        paths, labels = deepcopy(self.paths), deepcopy(self.y)

        if self.shuffle:
            paths, labels = sklearn.utils.shuffle(paths, labels)

        while True:
            yield self.load(paths[file_index]), self.get_target(labels[file_index])

            file_index = (file_index + 1) % len(paths)
            if file_index == 0:
                if not self.repeat:
                    break

                if self.shuffle:
                    paths, labels = sklearn.utils.shuffle(paths, labels)

    def __call__(self, buffer_size=0):
        dataset = tf.data.Dataset.from_generator(self.get_data, output_signature=(
            tf.TensorSpec(shape=self.X_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.y_shape, dtype=tf.float32)
        ))

        if self.shuffle and buffer_size > 0:
            if buffer_size > len(self):
                buffer_size = len(self)
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(self.batch_size)
        if self.repeat:
            dataset = dataset.repeat()

        return dataset

    def __len__(self):
        return len(self.y)//self.batch_size
