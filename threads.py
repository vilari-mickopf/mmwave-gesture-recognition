#!/usr/bin/env python

import time
import queue
import threading
from abc import ABC, abstractmethod

import numpy as np

from mmwave_gesture.data import Formats, GESTURE
from mmwave_gesture.data.preprocessor import ConsistentShape

import colorama
from colorama import Fore
colorama.init(autoreset=True)


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class Thread(ABC, threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()

        self.lock = threading.Lock()
        self._forward_list = []
        self.depends = []

        self.stop_event = threading.Event()

    @property
    def depends(self):
        with self.lock:
            return self._depends

    @depends.setter
    def depends(self, value):
        with self.lock:
            self._depends = value

    @property
    def forward_list(self):
        with self.lock:
            return self._forward_list

    def forward_to(self, thread):
        if thread not in self.forward_list:
            self.forward_list.append(thread)

        if self not in thread.depends:
            thread.depends.append(self)

    def forward(self, data):
        for t in self.forward_list.copy():
            if t.is_alive():
                t.queue.put(data)
            else:
                self.forward_list.remove(t)

    def collect(self):
        try:
            return self.queue.get(True, 1)
        except queue.Empty:
            return

    def stop(self):
        self.stop_event.set()

    def run(self):
        while True:
            for thread in self.depends:
                if not thread.is_alive():
                    self.stop()

            if self.stop_event.is_set():
                return

            self.process()

    @abstractmethod
    def process(self):
        pass


class ListenThread(Thread):
    def __init__(self, mmwave):
        super().__init__()
        self.mmwave = mmwave

    def process(self):
        data = self.mmwave.get_data()
        self.forward(data)
        time.sleep(.01)


class ParseThread(Thread):
    def __init__(self, parser, prompt, convert_objs=True):
        super().__init__()
        self.parser = parser
        self.prompt = prompt
        self.convert_objs = convert_objs
        self.sync = False

    def process(self):
        data = self.collect()
        frames = self.parser.assemble(data)
        if self.sync != self.parser.sync:
            print(self.prompt, end='')
            self.sync = self.parser.sync

        if frames is None:
            return

        for frame in frames.split(Formats.MAGIC_NUMBER):
            if not frame:
                continue

            frame = self.parser.parse(Formats.MAGIC_NUMBER + frame, warn=True)
            if (self.convert_objs and frame and
                    frame.get('tlvs', {}).get('detectedPoints')):
                self.parser.convert_detected_points(frame['tlvs']['detectedPoints'])

            self.forward(frame)


class PrintThread(Thread):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser

    def process(self):
        frame = self.collect()
        self.parser.pprint(frame)


class PredictThread(Thread):
    def __init__(self, model, logger, debug=False):
        super().__init__()
        self.model = model
        self.logger = logger
        self.debug = debug

        self.max_frames = None
        if self.model.preprocessor is not None:
            for p in self.model.preprocessor:
                if isinstance(p, ConsistentShape):
                    self.max_frames = p.num_of_frames

    def process(self):
        frame = self.collect()
        data = self.logger.log(frame, max_frames=self.max_frames, echo=False)
        if data is None or not self.logger.check_len(data):
            return

        pred = self.model.predict(data)
        if pred is None:
            if self.debug:
                print(f'{Fore.YELLOW}All points are outside of range')
            return

        gesture = GESTURE[np.argmax(pred)]
        if gesture != GESTURE.NONE:
            print(f'{Fore.GREEN}Gesture recognized: {Fore.BLUE}{gesture.name}')
            print(f'{Fore.CYAN}{"="*30}\n')

        if self.debug:
            for idx in np.argsort(pred)[::-1]:
                print(f'{Fore.YELLOW}{GESTURE[idx].name}: {pred[idx]*100:.2f}%')
            print(f'{Fore.CYAN}{"="*30}\n')


class LogThread(Thread):
    def __init__(self, logger, gesture):
        super().__init__()
        self.logger = logger
        self.gesture = gesture

    def process(self):
        frame = self.collect()
        data = self.logger.log(frame, echo=True)
        if data is not None:
            self.logger.save(data, self.gesture)
            self.stop()


class PlotThread(Thread):
    def __init__(self, plotter, main_queue):
        super().__init__()
        self.main_queue = main_queue
        self.plotter = plotter

        self.send_to_main(self.plotter.init)
        self.send_to_main(self.plotter.show)

    def stop(self):
        super().stop()
        self.send_to_main(self.plotter.close)

    def send_to_main(self, func, *args, **kwargs):
        self.main_queue.put((func, args, kwargs))

    def process(self):
        frame = self.collect()
        if frame is None:
            return

        self.send_to_main(self.plotter.plot_detected_objs, frame=frame)
