#!/usr/bin/env python

import os
import glob
import time

import numpy as np

from mmwave_gesture.data import GESTURE, DataLoader
from mmwave_gesture.utils.prints import print, warning

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Logger:
    def __init__(self, timeout=.5, data_dir=None):
        self.start_timeout = 3*timeout
        self.data_dir = data_dir
        self.end_timeout = timeout
        self.reset()

    def reset(self):
        self.data = None
        self.detected_time = 0
        self.empty_frames_cnt = 0
        self.timeout = self.start_timeout

    def log(self, frame, echo=False, max_frames=None):
        if self.data is None:
            self.data = []
            self.detected_time = time.perf_counter()
            self.timeout = self.start_timeout
            self.empty_frames_cnt = 0
            if echo:
                print('Saving sample...')

        if time.perf_counter() - self.detected_time > self.timeout:
            data = self.data
            self.reset()
            return data

        if frame and frame.get('tlvs', {}).get('detectedPoints'):
            self.timeout = self.end_timeout
            self.detected_time = time.perf_counter()

            if self.data:
                self.data.extend([None]*self.empty_frames_cnt)

            self.data.append(frame['tlvs']['detectedPoints']['objs'])
            if max_frames is not None and len(self.data) >= max_frames:
                data = self.data
                self.reset()
                return data

            self.empty_frames_cnt = 0
            return None

        self.empty_frames_cnt += 1
        return None

    def check_len(self, sample, echo=True):
        if sum(1 for frame in sample if frame is not None) < 3:
            if echo and not all(frame == None for frame in sample):
                warning('Gesture too short.\n')
            return False
        return True

    @staticmethod
    def get_gesture(gesture, dir):
        gesture = gesture if isinstance(gesture, GESTURE) else GESTURE[gesture]
        if dir is not None:
            gesture.dir = dir
        return gesture

    def save(self, data, gesture):
        if not data:
            warning('Nothing to save.\n')
            return

        if not self.check_len(data):
            return

        gesture = self.get_gesture(gesture, self.data_dir)
        if not os.path.exists(gesture.dir):
            os.makedirs(gesture.dir)

        np.savez_compressed(gesture.next_file(), data=np.array(data, dtype=object))
        print(f'{Fore.GREEN}Sample saved.\n')

    def discard_last_sample(self, gesture):
        last_sample = self.get_gesture(gesture, self.data_dir).last_file()
        if last_sample is None:
            print('No files.')
            return

        os.remove(last_sample)
        print('File deleted.')

    @staticmethod
    def get_data(gesture=None):
        X_paths, y = Logger.get_paths(gesture)
        X = [DataLoader(path).load() for path in X_paths]
        return X, y

    @staticmethod
    def _get_paths(gesture):
        paths, y = [], []
        if not os.path.exists(gesture.dir):
            return paths, y

        for f in glob.glob(f'{gesture.dir}/**/*.npz', recursive=True):
            paths.append(f)
            y.append(gesture.value)

        return paths, y

    @staticmethod
    def get_paths(gesture=None, dir=None):
        if gesture is not None:
            return Logger._get_paths(Logger.get_gesture(gesture, dir))

        # Get all gestures instead
        paths, y = [], []
        for gesture in GESTURE:
            gesture_paths, labels = Logger._get_paths(Logger.get_gesture(gesture, dir))
            if not gesture_paths or not labels:
                continue

            paths.extend(gesture_paths)
            y.extend(labels)

        return paths, y
