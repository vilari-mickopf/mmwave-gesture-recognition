#!/usr/bin/env python

from copy import deepcopy

import sklearn
import tensorflow as tf
import numpy as np

from mmwave.data.formats import GESTURE


class PolarPreprocessor:
    def __init__(self, formats, num_of_objs=65, num_of_frames=50):
        self.formats = formats
        self.num_of_objs = num_of_objs
        self.num_of_frames = num_of_frames

        self.empty = {'rho': 0., 'theta': 0., 'doppler': 0., 'peak': 0.}

    def process_obj(self, obj):
        x, y = obj['x'], obj['y']
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        if rho <= 1:
            return {
                'rho': rho, # already in range 0-1
                'theta': theta/np.pi,
                'doppler': obj['doppler']/self.formats.max_velocity,
                'peak': obj['peak']/(10*np.log10(1 + 2**16))
            }

        return None

    def zero_pad_objs(self, data):
        return tf.keras.preprocessing.sequence.pad_sequences(
            data, maxlen=self.num_of_objs, dtype='float32',
            padding='post', value=list(self.empty.values()))

    def zero_pad_frames(self, data):
        padding = np.tile(
            np.array([list(self.empty.values())]*self.num_of_objs)[np.newaxis, :, :],
            (self.num_of_frames - data.shape[0], 1, 1))

        return np.concatenate((data, padding), axis=0)

    def process(self, sample):
        data = []
        for frame in sample:
            objs = [list(processed_obj.values()) for obj in (frame or [])
                    for processed_obj in [self.process_obj(obj)] if processed_obj]

            if not objs:
                objs.append(list(self.empty.values()))

            if len(objs) > self.num_of_objs:
                objs = objs[:self.num_of_objs]

            data.append(objs)

        data = self.zero_pad_objs(data)
        if len(data) > self.num_of_frames:
            # print('Warning?')
            data = data[:self.num_of_frames]

        data = self.zero_pad_frames(data)
        return data


class DataGenerator:
    def __init__(self, paths, y, preprocessor=None, batch_size=32,
                       shuffle=False, repeat=False):
        self.X_paths = paths
        self.y = y
        self.preprocessor = preprocessor

        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle

        self.X_shape = self.load(self.X_paths[0]).shape
        self.y_shape = self.get_target(y[0]).shape

    def load(self, path, preprocess=True):
        data = np.load(path, allow_pickle=True)['data']
        if preprocess and self.preprocessor is not None:
            data = self.preprocessor.process(data)

        return data

    def get_target(self, label):
        return tf.keras.utils.to_categorical(label, num_classes=len(GESTURE))

    def get_data(self):
        file_index = 0
        paths, labels = deepcopy(self.X_paths), deepcopy(self.y)

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
