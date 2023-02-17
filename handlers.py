#! /usr/bin/env python

import os
import glob
import time
import readline

from signal import signal, SIGINT

from pynput.keyboard import Key, Controller


class SignalHandler:
    def __init__(self, queue):
        self.keyboard = Controller()
        self.signal_cnt = 0
        self.max_signal_cnt = 3
        self.detected_time = 0
        self.timeout = 1
        self.queue = queue

        signal(SIGINT, self.__ctrl_c_handler)

    def __ctrl_c_handler(self, signal_received, frame):
        if time.perf_counter() - self.detected_time < self.timeout:
            self.signal_cnt += 1
            if self.signal_cnt >= self.max_signal_cnt:
                os._exit(1)
        else:
            self.signal_cnt = 0

        self.queue.put(signal_received)

        self.detected_time = time.perf_counter()

        self.keyboard.press(Key.ctrl.value)
        self.keyboard.press('u')
        self.keyboard.release('u')

        self.keyboard.press(Key.enter)
        self.keyboard.release(Key.enter)


class Completer(object):
    def __init__(self, list):
        def list_completer(text, state):
            line = readline.get_line_buffer()
            if not line:
                return [c + ' ' for c in list][state]
            else:
                return [c + ' ' for c in list if c.startswith(line)][state]
        self.list_completer = list_completer

    def path_completer(self, text, state):
        line = readline.get_line_buffer().split()
        return [x for x in glob.glob(text + '*')][state]
