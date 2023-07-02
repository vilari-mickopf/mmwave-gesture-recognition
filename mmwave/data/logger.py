#!/usr/bin/env python

import os
import time

import numpy as np
from tqdm import tqdm

from multimethod import multimethod

from mmwave.data.formats import GESTURE
from mmwave.utils.prints import print, warning


class Logger:
    def __init__(self, timeout=.5):
        self.timeout = timeout
        self.reset()

    def reset(self):
        self.data = None
        self.detected_time = -1
        self.empty_frames_cnt = -1

    def log(self, frame):
        if self.data is None:
            self.data = []
            self.detected_time = time.perf_counter(),
            self.empty_frames_cnt = 0
            print(f'Saving sample...')

        if frame and frame.get('tlvs', {}).get('detectedPoints') is not None:
            self.detected_time = time.perf_counter()

            empty_frames = [[None]*len(frame)]*self.empty_frames_cnt
            self.data.extend(empty_frames)
            self.data.append(frame)

            return None

        self.empty_frames_cnt += 1
        if time.perf_counter() - self.detected_time > self.timeout:
            data = self.data
            self.reset()
            return data

    def save(self, gesture, data):
        gesture = gesture if isinstance(gesture, GESTURE) else GESTURE[gesture]
        if data is None or len(data) < 0:
            warning('Nothing to save.')

        np.savez_compressed(self.gesture.next_file(), data=self.data)

    def discard_last_sample(self):
        last_sample = self.gesture.last_file()
        if last_sample is None:
            print('No files.')
            return

        os.remove(last_sample)
        print('File deleted.')

    @staticmethod
    @multimethod
    def get_data(gesture):
        if not isinstance(gesture, GESTURE):
            gesture = GESTURE[gesture]

        for f in tqdm(os.listdir(gesture.dir), desc='Files', leave=False):
            yield np.load(os.path.join(gesture.dir, f), allow_pickle=True)['data']

            #  num_of_frames = df.iloc[-1]['frame'] + 1
            #  sample = [[] for _ in range(num_of_frames)]

            #  for _, row in df.iterrows():
                #  if row['x'] == 'None':
                    #  obj = 5*[0.]
                #  else:
                    #  obj = [
                        #  float(row['x'])/65535.,
                        #  float(row['y'])/65535.,
                        #  float(row['range_idx'])/65535.,
                        #  float(row['peak_value'])/65535.,
                        #  float(row['doppler_idx'])/65535.
                    #  ]
                #  sample[row['frame']].append(obj)

            #  yield sample

    @staticmethod
    @multimethod
    def get_data():
        X, y = [], []
        for gesture in tqdm(GESTURE, desc='Gestures'):
            for sample in Logger.get_data(gesture):
                X.append(sample)
                y.append(gesture.value)
        return X, y

    @staticmethod
    def get_stats(X, y):
        num_of_classes = len(set(y))
        print(f'Number of classes: {num_of_classes}')
        sample_with_max_num_of_frames = max(X, key=lambda sample: len(sample))

        max_num_of_frames = len(sample_with_max_num_of_frames)
        print(f'Maximum number of frames: {max_num_of_frames}')

        sample_with_max_num_of_objs = max(
            X, key=lambda sample: [len(frame) for frame in sample]
        )

        frame_with_max_num_of_objs = max(
            sample_with_max_num_of_objs, key=lambda obj: len(obj)
        )

        max_num_of_objs = len(frame_with_max_num_of_objs)
        print(f'Maximum num of objects: {max_num_of_objs}')
