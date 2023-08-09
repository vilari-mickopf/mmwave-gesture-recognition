#!/usr/bin/env python

from abc import ABC, abstractmethod

import numpy as np


class Preprocessor(ABC):
    @abstractmethod
    def process(self, sample):
        pass


class ZeroPadd(Preprocessor):
    def __init__(self, num_of_objs, num_of_frames):
        self.num_of_objs = num_of_objs
        self.num_of_frames = num_of_frames

    def num_of_points(self, lst):
        if not isinstance(lst[0], list):
            return len(lst)

        # Check first non-empty list
        return self.num_of_points(next((l for l in lst if l), None))

    def zero_pad_objs(self, data, padding):
        data = [sample + [padding]*(self.num_of_objs - len(sample)) for sample in data]
        return np.array(data)

    def zero_pad_frames(self, data, padding):
        padding = np.tile(
            np.array([padding]*self.num_of_objs)[np.newaxis, :, :],
            (self.num_of_frames - data.shape[0], 1, 1))

        return np.concatenate((data, padding), axis=0)

    def process(self, sample):
        padd = [0.]*self.num_of_points(sample)
        data = self.zero_pad_objs([frame[:self.num_of_objs] for frame in sample], padd)
        data = self.zero_pad_frames(data[:self.num_of_frames], padd)
        return data


class Polar(Preprocessor):
    def __init__(self, formats, max_distance=1):
        self.formats = formats
        self.max_distance = max_distance

    def process_obj(self, obj):
        x, y = obj['x'], obj['y']
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        if rho <= self.max_distance:
            # -max_velocity : max_velocity -> 0 : 2*max_velocity
            doppler = self.formats.max_velocity + obj['doppler']

            return {
                'rho': rho/self.max_distance,
                'theta': theta/np.pi,
                'doppler': doppler/2*self.formats.max_velocity,
                'peak': obj['peak']/(10*np.log10(1 + 2**16))
            }

        return None

    def process(self, sample):
        data = []
        for frame in sample:
            objs = [list(processed_obj.values()) for obj in (frame or [])
                    for processed_obj in [self.process_obj(obj)] if processed_obj]
            data.append(objs)

        return data


if __name__ == '__main__':
    import seaborn as sns

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-dark')

    from mmwave_gesture.data import Logger, DataLoader


    def hist(counts, name=''):
        sns.histplot(counts, bins=np.unique(counts))

        ax = plt.gca()
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2, p.get_height(),
                    str(int(p.get_height())), fontsize=7,
                    color='black', ha='center', va='bottom')

        bins = range(min(counts), max(counts) + 2)
        plt.xticks([i + .5 for i in bins[:-1]])
        plt.title(f'Histogram of {name}')
        ax.set_xticklabels(bins[:-1])
        ax.set_xlabel(name)
        plt.show()


    paths, y = Logger.get_paths()
    data = [DataLoader(path).load() for path in paths
            if 'none' not in path]

    frame_cnt = [len(sample) for sample in data]
    print(f'{max(frame_cnt)=}')
    hist(frame_cnt, 'frames')

    data = [DataLoader(path).load() for path in paths]
    obj_cnt = [len(frame) for sample in data
               for frame in sample if frame is not None]
    print(f'{max(obj_cnt)=}')
    hist(obj_cnt, 'objs')
