#!/usr/bin/env python

import time

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from mmwave.data.logger import Logger


# Global plot config
mpl.use('Qt5Agg')
mpl.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-dark')


class Plotter:
    def __init__(self, queue):
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

    def get_previos_data(self, gesture):
        last_sample = Logger.get_last_sample(gesture)
        if last_sample[-1] == '/':
            return None

        df = pd.read_csv(last_sample).reset_index().values

        num_of_frames = df[-1][1]+1
        points = [[] for _ in range(num_of_frames)]
        for row in df:
            if row[2] == 'None' or row[3] == 'None':
                points[row[1]].append((None, None))
            else:
                x, y = int(row[2]), int(row[3])
                if x > 32767:
                    x = x - 65536

                if y > 32767:
                    y = y - 65536

                x = x/int(row[7])
                y = y/int(row[7])

                points[row[1]].append((x, y))
        return points

    def draw_last_sample(self, gesture):
        self.sc.set_color([.5, .5, .5])
        last_sample = self.get_previos_data(gesture)
        print('Redrawing...')

        self.update()
        for frame in last_sample:
            self.update([[x, y] for x, y in frame])
            time.sleep(.03)

        print('Done')

        self.update()
        self.sc.set_color([1, 0, 0])

    def plot_detected_objs(self, frame):
        points = []
        if frame and frame.get('tlvs', {}).get('detectedPoints') is not None:
            detected_points = frame['tlvs']['detectedPoints']
            assert detected_points['descriptor'].get('converted') == True

            for obj in detected_points['objs']:
                if not obj or None in obj.values():
                    continue

                points.append([obj['x'], obj['y']])

        self.update(points)

    def show(self):
        plt.gcf().canvas.flush_events()
        plt.show(block=False)
        plt.gcf().canvas.flush_events()

    def close(self):
        plt.close()
