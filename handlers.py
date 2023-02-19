#! /usr/bin/env python

import sys
import os
import time
import readline

from signal import signal, SIGINT

from mmwave.utils.prints import warning


class SignalHandler:
    def __init__(self, queue, timeout=1):
        self.queue = queue
        self.timeout = timeout

        self.signal_cnt = 0
        self.max_signal_cnt = 3
        self.detected_time = -1

        self.exit_state = False
        signal(SIGINT, self.ctrl_c_handler)

    def check_timeout(self):
        is_timeout = True
        timeout = 3*self.timeout if self.exit_state else self.timeout
        if time.perf_counter() - self.detected_time < timeout:
            is_timeout = False
        self.detected_time = time.perf_counter()
        return is_timeout

    def ctrl_c_handler(self, signal_received, frame):
        if not self.check_timeout():
            if self.exit_state:
                os._exit(1)

            self.signal_cnt += 1
            if self.signal_cnt >= self.max_signal_cnt - 1:
                self.exit_state = True
                warning('\nPress <Ctrl-C> one more time to exit '
                        f'({3*self.timeout} sec left)\n')
        else:
            self.signal_cnt = 0
            self.exit_state = False

        self.queue.put(signal_received)


class Completer(object):
    def __init__(self, list):
        def list_completer(text, state):
            line = readline.get_line_buffer()
            if not line:
                return [f'{c} ' for c in list][state]
            else:
                return [f'{c} ' for c in list if c.startswith(line)][state]
        self.list_completer = list_completer
