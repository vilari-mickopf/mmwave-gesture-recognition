#! /usr/bin/env python

import time
import queue
import threading

from abc import ABC, abstractmethod


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
            for t in self.depends:
                if not t.is_alive():
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
        time.sleep(.1)


class ParseThread(Thread):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser

    def process(self):
        data = self.collect()
        frames = self.parser.assemble(data)
        if frames is None:
            return

        for frame in frames.split(self.parser.formats.MAGIC_NUMBER):
            if not frame:
                continue

            f = self.parser.parse(self.parser.formats.MAGIC_NUMBER+frame, warn=True)
            self.forward(f)


class PrintThread(Thread):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser

    def process(self):
        frame = self.collect()
        self.parser.pprint(frame)


class PredictThread(Thread):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.collecting = False
        self.sequence = []
        self.empty_frames = []
        self.detected_time = time.perf_counter()
        self.frame_num = 0

        self.num_of_data_in_obj = 5
        self.num_of_frames = 50

    def process(self):
        frame = self.collect()
        if not self.collecting:
            self.collecting = True
            self.sequence = []
            self.detected_time = time.perf_counter()

        if (frame is not None and
                frame.get('tlvs') is not None and
                frame['tlvs'].get(1) is not None):
            self.detected_time = time.perf_counter()
            if self.frame_num == 0:
                self.sequence = []

            for empty_frame in self.empty_frames:
                self.sequence.append(empty_frame)
                self.empty_frames = []

            objs = []
            for obj in frame['tlvs'][1]['values']['objs']:
                if obj is None or None in obj.values():
                    continue
                objs.append([
                    obj['x_coord']/65535.,
                    obj['y_coord']/65535.,
                    obj['range_idx']/65535.,
                    obj['peak_value']/65535.,
                    obj['doppler_idx']/65535.
                ])
            self.sequence.append(objs)
            self.frame_num += 1

            if self.frame_num >= self.num_of_frames:
                self.empty_frames = []
                self.collecting = False
                self.frame_num = 0

        elif self.frame_num != 0:
            self.empty_frames.append([[0.]*self.num_of_data_in_obj])
            self.frame_num += 1

        if time.perf_counter() - self.detected_time > .5:
            self.empty_frames = []
            self.collecting = False
            self.frame_num = 0

            if len(self.sequence) > 3:
                self.model.predict([self.sequence])


class LogThread(Thread):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def process(self):
        frame = self.collect()
        if self.logger.log(frame):
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
