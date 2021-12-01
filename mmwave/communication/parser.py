#! /usr/bin/env python

import time
import struct
from copy import deepcopy
from pprint import pformat

import pandas as pd

from mmwave.utils.utility_functions import print

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Parser:
    def __init__(self, formats):
        self.sync = False
        self.sync_time = 0
        self.sync_timeout = 2

        self.buffer = bytearray(b'')

        self.formats = formats

        self.parsing = False
        self.frame = None
        self.frame_num = 0

    def assemble(self, data):
        if data is None:
            return None

        start_idx = data.find(self.formats.MAGIC_NUMBER)
        if start_idx == 0:
            # Save return value, and start capturing new frame
            frame = deepcopy(self.buffer)
            self.buffer = bytearray(data)

            self.sync_time = time.perf_counter()
            if self.sync is False:
                print(f'{Fore.GREEN}Sync received!\n')
                self.sync = True
                return None
            return frame

        elif self.sync is False:
            if self.sync_time == 0:
                self.sync_time = time.perf_counter()
                print(f'{Fore.YELLOW}Waiting for sync...')

            # Capture long sync time
            if time.perf_counter() - self.sync_time > self.sync_timeout:
                print(f'{Fore.RED}No sync received.')
                print('Please check your board.\n')
                self.sync_time = 0

        else:
            # Wait for rest of the frame
            self.buffer.extend(bytearray(data))

            # Capture long no data time
            if data == b'':
                if time.perf_counter() - self.sync_time > self.sync_timeout:
                    print(f'{Fore.RED}Not receiving data. Resyncing...')
                    self.sync_time = 0
                    self.sync = False

    def parse(self, frame, warn=False):
        self.frame = frame

        header = self.__parse_struct(self.formats.header, echo=warn)
        if header is None:
            return None

        header = header['values']

        if not self.__len_check(header['packet_len'], len(frame), echo=warn):
            return None

        self.__header_frame_num_check(header, echo=warn)

        tlvs = {}
        for tlv in range(header['num_tlvs']):
            tlv_type = self.__parse_value('I', echo=warn)
            tlv_len = self.__parse_value('I', echo=warn)

            if self.formats.tlvs.get(tlv_type) is None:
                return None

            tlv_format = self.formats.tlvs[tlv_type]
            tlvs[tlv_type] = self.__parse_struct(tlv_format, echo=warn)

        return {'header': header, 'tlvs': tlvs}

    def __parse_struct(self, struct_format, echo=False):
        parsed_struct = {}
        for key, value_format in struct_format.items():
            if isinstance(value_format, dict):
                value = self.__parse_struct(value_format, echo)
            elif key == 'objs' and isinstance(value_format, list):
                size_idxs = value_format[0].split('.')
                size = parsed_struct[size_idxs[0]]
                for i in size_idxs[1:]:
                    size = size[i]

                value = []
                for obj in range(size):
                    value.append(self.__parse_struct(value_format[1], echo))
            else:
                if key == 'name':
                    value = value_format
                else:
                    value = self.__parse_value(value_format, echo)
            parsed_struct[key] = value
        return parsed_struct

    def __parse_value(self, value_format, echo=False):
        size = struct.calcsize(value_format)
        if not self.__len_check(len(self.frame), size, echo=echo):
            return None

        value = struct.unpack(value_format, self.frame[:size])
        self.frame = self.frame[size:]

        if len(value) == 1:
            value = value[0]

        if 's' in value_format:
            value = bytearray(value)
            value.reverse()
            value = ':'.join(format(x, '02x') for x in value)
        return value

    def __header_frame_num_check(self, header, echo=False):
        if self.frame_num != 0 and self.frame_num + 1 != header['frame_num']:
            if echo:
                num_of_missed_frames = header['frame_num'] - self.frame_num - 1
                print(f'{Fore.YELLOW}WARNING: Missed {num_of_missed_frames} frames.')
        self.frame_num = header['frame_num']

    def __len_check(self, received, expected, echo=False):
        if received < expected:
            if echo:
                print(f'{Fore.RED}ERROR: Corrupted frame.')
            return False
        return True

    @staticmethod
    def pprint(frame):
        if frame is None:
            print(f'{Fore.MAGENTA}Frame: {Fore.RED}None')
            return

        print(Fore.CYAN + '='*85)
        Parser.__pprint_struct(frame)
        print(Fore.CYAN + '='*85)
        print()

    @staticmethod
    def __pprint_struct(frame, indentation='', _recursive_call=False):
        identetion_marker = '|' + 3*' '

        if not _recursive_call:
            Parser.__pprint_struct.num = 1

        for key, value in frame.items():
            if isinstance(value, dict):
                print(indentation, end='')
                color = Parser.__pprint_get_color(Parser.__pprint_struct.num - 1)
                print(f'{color}{key}:')

                indentation += identetion_marker
                Parser.__pprint_struct.num += 1
                Parser.__pprint_struct(value, indentation, True)
                Parser.__pprint_struct.num -= 1
                indentation = indentation[:-4]

                if key != sorted(frame.keys())[-1]:
                    print(indentation)
            elif isinstance(value, list):
                print(indentation, end='')
                color = Parser.__pprint_get_color(Parser.__pprint_struct.num - 1)
                print(f'{color}{key}:')

                indentation += identetion_marker
                for obj in value:
                    Parser.__pprint_struct.num += 1
                    Parser.__pprint_struct(obj, indentation, True)
                    Parser.__pprint_struct.num -= 1
                    if obj != value[-1]:
                        print(indentation)
                indentation = indentation[:-4]
            else:
                if isinstance(value, tuple):
                    print(indentation, end='')
                    color = Parser.__pprint_get_color(Parser.__pprint_struct.num - 1)
                    print(f'{color}{key}:')
                    indentation += identetion_marker

                    with pd.option_context('display.max_rows', 10,
                                           'display.max_columns', 6,
                                           'precision', 2,
                                           'show_dimensions', False):
                        df = pd.DataFrame(value)
                        df = pformat(df)
                        for line_idx, line in enumerate(df.split('\n')):
                            print(indentation, end='')
                            for word_idx, word in enumerate(line.split(' ')):
                                color = ''
                                if word_idx == 0 or line_idx == 0:
                                    color = Fore.YELLOW
                                print(f'{color}{word}', end=' ')
                            print()
                    indentation = indentation[:-4]
                else:
                    print(indentation, end='')
                    color = Parser.__pprint_get_color(Parser.__pprint_struct.num - 1)
                    print(f'{color}{key}: {value}')

    @staticmethod
    def __pprint_get_color(num_of_identations):
        if num_of_identations % 4 == 0:
            return Fore.BLUE
        elif num_of_identations % 3 == 0:
            return Fore.CYAN
        elif num_of_identations % 2 == 0:
            return Fore.GREEN
        else:
            return Fore.MAGENTA
