#!/usr/bin/env python

import threading

import colorama
from colorama import Fore
colorama.init(autoreset=True)


# Thread-safe print
_print = print
_print_lock = threading.Lock()
def print(*args, **kwargs):
    with _print_lock:
        _print(*args, **kwargs)


def error(msg, **kwargs):
    print(f'{Fore.RED}{msg}', **kwargs)


def warning(msg, **kwargs):
    print(f'{Fore.YELLOW}{msg}', **kwargs)
