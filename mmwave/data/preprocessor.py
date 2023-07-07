#!/usr/bin/env python

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class Preprocessor(ABC):
    @abstractmethod
    def process(self, sample):
        pass


class ZeroPadd(Preprocessor):
    def __init__(self, num_of_objs=65, num_of_frames=50):
        self.num_of_objs = num_of_objs
        self.num_of_frames = num_of_frames
        self.empty = {'rho': 0., 'theta': 0., 'doppler': 0., 'peak': 0.}

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
        data = self.zero_pad_objs([frame[:self.num_of_objs] for frame in sample])
        data = self.zero_pad_frames(data[:self.num_of_frames])
        return data


class Polar(Preprocessor):
    def __init__(self, formats):
        self.formats = formats
        self.empty = {'rho': 0., 'theta': 0., 'doppler': 0., 'peak': 0.}

    def process_obj(self, obj):
        x, y = obj['x'], obj['y']
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)
        if rho <= 1:
            return {
                'rho': rho, # already in range 0-1
                'theta': theta/(np.pi/2),
                'doppler': obj['doppler']/self.formats.max_velocity,
                'peak': obj['peak']/(10*np.log10(1 + 2**16))
            }

        return None

    def process(self, sample):
        data = []
        for frame in sample:
            objs = [list(processed_obj.values()) for obj in (frame or [])
                    for processed_obj in [self.process_obj(obj)] if processed_obj]

            if not objs:
                objs.append(list(self.empty.values()))

            data.append(objs)

        return data
