#!/usr/bin/env python

import time

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from mmwave_gesture.data import GESTURE, DataLoader

import colorama
from colorama import Fore
colorama.init(autoreset=True)


# Global plot config
mpl.use('Qt5Agg')
mpl.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-v0_8-dark')


class Plotter:
    def __init__(self, queue=None):
        self.queue = queue

        self.fig = None
        self.ax = None
        self.sc = None

        #  self.init()  # Make sure this is started from main thread

    def init(self):
        plt.close('all')
        self.set_figure()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        connect = self.fig.canvas.mpl_connect
        self.draw_cid = connect('draw_event', self.grab_background)
        connect('close_event', self.fig_close)

    def set_figure(self):
        self.fig, self.ax = plt.subplots()
        self.sc = self.ax.scatter([], [], color='r', picker=10, animated=True)

        # Ticks with resolution of 25cm
        self.ax.grid(False)
        plt.xticks(np.arange(-1, 1.01, step=.25),
                   ('-1m', '-0.75m', '-0.5m', '-0.25m',
                      '0', '+0.25m', '+0.5m', '+0.75m', '1m'))
        plt.yticks(np.arange(0, 1.251, step=.25),
                   ('0', '0.25m', '0.5m', '0.75m', '1m', '1.25m'))
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(0, 1.41)

        # Plot guidelines
        for r in np.linspace(0, 2, 9):
            x_circle = r * np.cos(np.linspace(0, np.pi, 100))
            y_circle = r * np.sin(np.linspace(0, np.pi, 100))
            self.ax.plot(x_circle, y_circle, color='white', linewidth=1)

        x_diagonal = np.linspace(-2, 2, 10)
        self.ax.plot(x_diagonal, x_diagonal, color='white', linewidth=1)
        self.ax.plot(x_diagonal, -x_diagonal, color='white', linewidth=1)
        self.ax.axvline(x=0, color='white', linewidth=1)

        self.sc.set_color([1, 0, 0])

    def grab_background(self, event=None):
        self.sc.set_visible(False)

        # Temporarily disconnect the draw_event callback to avoid recursion
        canvas = self.fig.canvas
        canvas.mpl_disconnect(self.draw_cid)
        canvas.draw()

        self.draw_cid = canvas.mpl_connect('draw_event', self.grab_background)
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.sc.set_visible(True)

    def fig_close(self, event=None):
        if self.queue is not None:
            self.queue.put('closed')

    def blit(self):
        self.fig.canvas.restore_region(self.background)
        self.ax.draw_artist(self.sc)
        self.fig.canvas.blit(self.ax.bbox)

    def update(self, points=None):
        if not points:
            points = [None, None]

        self.sc.set_offsets(points)
        self.blit()
        plt.gcf().canvas.flush_events()

    def plot_sample(self, sample, color=[0, .5, 1]):
        self.sc.set_color(color)

        print('Redrawing...', end='')
        self.update()
        for frame in sample:
            objs = None if frame is None else [[obj['x'], obj['y']] for obj in frame]
            self.update(objs)
            time.sleep(.066)

        self.update()
        self.sc.set_color([1, 0, 0])
        print('Done.\n')

    def plot_detected_objs(self, frame):
        points = None
        if frame and frame.get('tlvs', {}).get('detectedPoints') is not None:
            detected_points = frame['tlvs']['detectedPoints']
            assert detected_points['descriptor'].get('converted') == True

            points = [[obj['x'], obj['y']] for obj in detected_points['objs']
                      if not obj or None not in obj.values()]

        self.update(points)

    def show(self):
        plt.gcf().canvas.flush_events()
        plt.show(block=False)
        plt.gcf().canvas.flush_events()

    def close(self):
        plt.close(self.fig)


if __name__ == '__main__':
    gesture = GESTURE.CCW

    last_file = gesture.last_file()
    if last_file is None:
        print(f'{Fore.YELLOW}No samples for gesture {gesture.name}.')
        exit()

    plotter = Plotter()
    plotter.init()
    plotter.show()
    plotter.plot_sample(DataLoader(last_file).load())
    plotter.close()
