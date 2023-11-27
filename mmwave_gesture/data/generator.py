#!/usr/bin/env python

import os
from copy import deepcopy

import numpy as np
import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from mmwave_gesture.data import GESTURE
from mmwave_gesture.data import DataLoader


class DataGenerator:
    def __init__(self, paths, y, preprocessor=None, batch_size=1,
                       shuffle=False, repeat=False, loader=DataLoader()):
        self.paths = paths
        self.loader = loader
        self.y = y

        self.preprocessor = preprocessor
        if preprocessor is not None and not isinstance(preprocessor, list):
            self.preprocessor = [preprocessor]

        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle

        X_sample = self.loader.load(self.paths[0])
        self.X_shape = self.preprocess(X_sample, self.preprocessor).shape
        self.y_shape = self.get_target(y[0]).shape

    @staticmethod
    def preprocess(data, preprocessor):
        if preprocessor is not None:
            for p in preprocessor:
                data = p.transform(data)
                if data is None:
                    return None

        return np.array(data)

    def get_target(self, label):
        return tf.keras.utils.to_categorical(label, num_classes=len(GESTURE))

    def get_data(self):
        file_index = 0
        paths, labels = deepcopy(self.paths), deepcopy(self.y)

        if self.shuffle:
            paths, labels = sklearn.utils.shuffle(paths, labels)

        while True:
            X = self.loader.load(paths[file_index])
            y = self.get_target(labels[file_index])

            X = self.preprocess(X, self.preprocessor)
            if X is not None:
                yield X, y

            file_index = (file_index + 1) % len(paths)
            if file_index == 0:
                if not self.repeat:
                    break

                if self.shuffle:
                    paths, labels = sklearn.utils.shuffle(paths, labels)

    def __call__(self):
        dataset = tf.data.Dataset.from_generator(self.get_data, output_signature=(
            tf.TensorSpec(shape=self.X_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.y_shape, dtype=tf.float32)
        ))

        dataset = dataset.cache()

        dataset = dataset.batch(self.batch_size)
        if self.repeat:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        # Some samples can yield None, so we have to count them manually
        gen = DataGenerator(self.paths, self.y, preprocessor=self.preprocessor,
                            loader=self.loader)
        num_of_samples = sum(1 for _ in gen.get_data())
        return num_of_samples//self.batch_size
