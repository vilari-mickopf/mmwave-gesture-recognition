#!/usr/bin/env python

import os

import numpy as np

from mmwave_gesture.data import Logger, DataLoader, GESTURE
from mmwave_gesture.model import *
from mmwave_gesture.utils import Plotter

import cleanlab


def show(model, paths):
    plotter = Plotter()
    plotter.init()
    plotter.show()
    for path in paths:
        if not os.path.exists(path):
            continue

        sample = DataLoader(path).load()
        print(f'Prediction for path {path}:')
        for i, p in enumerate(np.array(model.predict(sample))*100):
            if int(p) > 0:
                print(f'{GESTURE[i].name}: {int(p)}%')

        finished = False
        while not finished:
            plotter.plot_sample(sample)
            user_input = input('[n]ext/[d]elete')
            if user_input in ['n', 'next']:
                finished = True
            elif user_input in ['d', 'delete']:
                print(f'Removing {path}')
                os.remove(path)
                finished = True
            else:
                print(f'Redrawing {path}')


if __name__ == '__main__':
    paths, y = Logger.get_paths()

    config = os.path.join(os.path.dirname(__file__),
                          'mmwave/communication/profiles/profile.cfg')

    model = Conv1DModel()
    model.load()
    print('Evaluating...', end='', flush=True)
    y_true, y_pred = model.evaluate(paths, y)
    print('Done...')

    print('Finding issues...', end='', flush=True)
    label_issues = cleanlab.filter.find_label_issues(
        y_true, np.array(y_pred), return_indices_ranked_by='self_confidence')
    print('Done...')

    show(model, [paths[i] for i in label_issues])
    del model
