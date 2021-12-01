#! /usr/bin/env python

import os
import time
import pickle

import pandas as pd
from tqdm import tqdm

from mmwave.data.formats import GESTURE
from mmwave.utils.utility_functions import print

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Logger:
    def __init__(self, gesture=None):
        self.logging = False
        self.gesture = gesture
        self.log_file = ''
        self.detected_time = 0
        self.empty_frames = ''
        self.frame_num = 0

    def __set_file(self):
        last_sample = self.get_last_sample(self.gesture)
        if last_sample is None:
            self.log_file = os.path.join(last_sample, 'sample_1.csv')
            return

        save_dir = os.path.dirname(last_sample)
        last_sample_name = os.path.splitext(last_sample)[0]
        num = int(os.path.basename(last_sample_name).split('_')[1]) + 1
        self.log_file = os.path.join(save_dir, 'sample_' + str(num) + '.csv')
        print(f'Sample number: {num}')

    def set_gesture(self, gesture):
        self.gesture = gesture

    def log(self, frame):
        if not self.logging:
            self.__set_file()
            self.logging = True
            self.detected_time = time.perf_counter()
            print('Saving...')

        if (frame is not None and
                frame.get('tlvs') is not None and
                frame['tlvs'].get(1) is not None):
            self.detected_time = time.perf_counter()
            with open(self.log_file, 'a') as f:
                if self.frame_num == 0:
                    f.write('frame,x,y,range_idx,peak_value,doppler_idx,xyz_q_format\n')

                for obj in frame['tlvs'][1]['values']['objs']:
                    f.write(self.empty_frames)
                    f.write(str(self.frame_num) + ',')
                    f.write(str(obj['x_coord']) + ',')
                    f.write(str(obj['y_coord']) + ',')
                    f.write(str(obj['range_idx']) + ',')
                    f.write(str(obj['peak_value']) + ',')
                    f.write(str(obj['doppler_idx']) + ',')
                    f.write(str(frame['tlvs'][1]['values']['descriptor']['xyz_q_format']) + '\n')
                    self.empty_frames = ''

            self.frame_num += 1

        elif self.frame_num != 0:
            self.empty_frames = (self.empty_frames +
                                   str(self.frame_num) + ',')
            self.empty_frames = (self.empty_frames +
                                   'None, None, None, None, None\n')
            self.frame_num += 1

        if time.perf_counter() - self.detected_time > .5:
            if os.path.isfile(self.log_file):
                print('Sample saved.\n')
            else:
                print('Nothing to save.\n')
            self.empty_frames = ''

            self.logging = False
            self.frame_num = 0
            return True

        return False

    @staticmethod
    def get_last_sample(gesture):
        if isinstance(gesture, str):
            gesture = GESTURE[gesture.upper()]
        save_dir = gesture.get_dir()

        if os.listdir(save_dir) == []:
            return

        nums = []
        for f in os.listdir(save_dir):
            num = os.path.splitext(f)[0].split('_')[1]
            nums.append(int(num))
        last_sample = 'sample_' + str(max(nums)) + '.csv'

        return os.path.join(save_dir, last_sample)

    def discard_last_sample(self):
        last_sample = self.get_last_sample(self.gesture)
        if last_sample is None:
            print('No files.')
            return

        os.remove(last_sample)
        print('File deleted.')

    @staticmethod
    def get_data(gesture):
        if isinstance(gesture, str):
            gesture = GESTURE[gesture.upper()]
        save_dir = gesture.get_dir()
        for f in tqdm(os.listdir(save_dir), desc='Files', leave=False):
            df = pd.read_csv(os.path.join(save_dir, f))
            num_of_frames = df.iloc[-1]['frame'] + 1
            sample = [[] for _ in range(num_of_frames)]

            for _, row in df.iterrows():
                if row['x'] == 'None':
                    obj = 5*[0.]
                else:
                    obj = [
                        float(row['x'])/65535.,
                        float(row['y'])/65535.,
                        float(row['range_idx'])/65535.,
                        float(row['peak_value'])/65535.,
                        float(row['doppler_idx'])/65535.
                    ]
                sample[row['frame']].append(obj)

            yield sample

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

        return max_num_of_frames, max_num_of_objs, num_of_classes

    @staticmethod
    def get_all_data(refresh_data=False):
        X_file = os.path.join(os.path.dirname(__file__), '.X_data')
        y_file = os.path.join(os.path.dirname(__file__), '.y_data')
        if refresh_data:
            X = []
            y = []
            for gesture in tqdm(GESTURE, desc='Gestures'):
                for sample in Logger.get_data(gesture):
                    X.append(sample)
                    y.append(gesture.value)
            pickle.dump(X, open(X_file, 'wb'))
            pickle.dump(y, open(y_file, 'wb'))
        else:
            print('Loading cached data...', end='')
            X = pickle.load(open(X_file, 'rb'))
            y = pickle.load(open(y_file, 'rb'))
            print(f'{Fore.GREEN}Done.')
        return X, y
