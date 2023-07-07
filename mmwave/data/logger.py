#!/usr/bin/env python

import os
import glob
import time

import numpy as np

from mmwave.data import GESTURE
from mmwave.utils.prints import print, warning

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Logger:
    def __init__(self, timeout=.5):
        self.start_timeout = 3*timeout
        self.end_timeout = timeout
        self.reset()

    def reset(self):
        self.data = None
        self.detected_time = 0
        self.empty_frames_cnt = 0
        self.timeout = self.start_timeout

    def log(self, frame, echo=False):
        if self.data is None:
            self.data = []
            self.detected_time = time.perf_counter()
            self.timeout = self.start_timeout
            self.empty_frames_cnt = 0
            if echo:
                print('Saving sample...')

        if frame and frame.get('tlvs', {}).get('detectedPoints'):
            self.timeout = self.end_timeout
            self.detected_time = time.perf_counter()

            if self.data:
                self.data.extend([None]*self.empty_frames_cnt)

            self.data.append(frame['tlvs']['detectedPoints']['objs'])
            self.empty_frames_cnt = 0
            return None

        self.empty_frames_cnt += 1
        if time.perf_counter() - self.detected_time > self.timeout:
            data = self.data
            self.reset()
            return data

        return None

    @staticmethod
    def save(gesture, data):
        gesture = gesture if isinstance(gesture, GESTURE) else GESTURE[gesture]
        if not data:
            warning('Nothing to save.\n')
            return

        if sum(1 for frame in data if frame is not None) <= 3:
            warning('Sample too short.\n')
            return

        if not os.path.exists(gesture.dir):
            os.makedirs(gesture.dir)

        np.savez_compressed(gesture.next_file(), data=np.array(data, dtype=object))
        print(f'{Fore.GREEN}Sample saved.\n')

    @staticmethod
    def discard_last_sample(gesture):
        last_sample = gesture.last_file()
        if last_sample is None:
            print('No files.')
            return

        os.remove(last_sample)
        print('File deleted.')

    @staticmethod
    def get_data(gesture=None):
        X_paths, y = Logger.get_paths(gesture)
        X = [np.load(path, allow_pickle=True)['data'] for path in X_paths]
        return X, y

    @staticmethod
    def _get_paths(gesture):
        if not isinstance(gesture, GESTURE):
            gesture = GESTURE[gesture]

        paths, y = [], []
        if not os.path.exists(gesture.dir):
            return paths, y

        for f in glob.glob(f'{gesture.dir}/**/*.npz', recursive=True):
            paths.append(f)
            y.append(gesture.value)

        return paths, y

    @staticmethod
    def get_paths(gesture=None):
        if gesture is not None:
            return Logger._get_paths(gesture)

        # Get all gestures instead
        paths, y = [], []
        for gesture in GESTURE:
            gesture_paths, labels = Logger._get_paths(gesture)
            if not gesture_paths or not labels:
                continue

            paths.extend(gesture_paths)
            y.extend(labels)

        return paths, y

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
