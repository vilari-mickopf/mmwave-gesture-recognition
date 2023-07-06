#!/usr/bin/env python

import time
import struct
import pprint
from copy import deepcopy

import numpy as np
import pandas as pd

from mmwave.utils.prints import print

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Parser:
    def __init__(self, formats):
        self.formats = formats

        self.parsing = False
        self.sync = False
        self.sync_time = -1
        self.sync_timeout = 2

        self.buffer = bytearray()

        self.frame = None
        self.frame_num = 0

    def assemble(self, data):
        if data is None:
            return None

        if data.find(self.formats.MAGIC_NUMBER) == 0:
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
            if self.sync_time == -1:
                self.sync_time = time.perf_counter()
                print(f'{Fore.YELLOW}Waiting for sync...')

            # Capture long sync time
            if time.perf_counter() - self.sync_time > self.sync_timeout:
                print(f'{Fore.RED}No sync received.')
                print('Please check your board.\n')
                self.sync_time = -1

        else:
            # Wait for rest of the frame
            self.buffer += bytearray(data)

            # Capture long no data time
            if not data and ((time.perf_counter() - self.sync_time) > self.sync_timeout):
                print(f'{Fore.RED}Not receiving data. Resyncing...')
                self.sync_time = -1
                self.sync = False

    def parse(self, frame, warn=False):
        self.frame = frame

        header = self.parse_struct(self.formats.header, echo=warn)
        if header is None:
            return

        if not self.len_check(header['packet_len'], len(frame), echo=warn):
            return

        if header['num_tlvs'] is None:
            if warn:
                print(f'{Fore.RED}ERROR: Corrupted frame.')
            return

        self.header_frame_num_check(header, echo=warn)

        tlvs = {}
        for tlv in range(header['num_tlvs']):
            tlv_index = int(self.parse_value('I', echo=warn)) - 1
            tlv_len = self.parse_value('I', echo=warn)

            if len(self.formats.tlvs) < tlv_index:
                if warn:
                    print(f'{Fore.RED}ERROR: Unknown tlv.')
                return

            tlv_type = list(self.formats.tlvs)[tlv_index]
            tlv = self.parse_struct(self.formats.tlvs[tlv_type], echo=warn)
            if tlv is None:
                return

            tlvs[tlv_type] = tlv

        return {'header': header, 'tlvs': tlvs}

    def parse_struct(self, struct_format, echo=False):
        if isinstance(struct_format, dict):
            parsed_struct = {}
            for key, value_format in struct_format.items():
                if isinstance(value_format, dict):
                    value = self.parse_struct(value_format, echo)

                elif isinstance(value_format, list):
                    size_idxs = value_format[0].split('.')
                    size = parsed_struct[size_idxs[0]]
                    for i in size_idxs[1:]:
                        size = size[i]

                    value = [self.parse_struct(value_format[1], echo)
                             for _ in range(size)]
                else:
                    value = self.parse_value(value_format, echo)

                if value is None:
                    return

                parsed_struct[key] = value
        else:
            parsed_struct = self.parse_value(struct_format, echo)

        return parsed_struct

    def parse_value(self, value_format, echo=False):
        size = struct.calcsize(value_format)
        if not self.len_check(len(self.frame), size, echo=echo):
            return

        value = struct.unpack(value_format, self.frame[:size])
        self.frame = self.frame[size:]

        if len(value) == 1:
            value = value[0]

        if 's' in value_format:
            value = bytearray(value)
            value.reverse()
            value = ':'.join(format(x, '02x') for x in value)
        return value

    def header_frame_num_check(self, header, echo=False):
        if self.frame_num != 0 and self.frame_num + 1 != header['frame_num']:
            if echo:
                num_of_missed_frames = header['frame_num'] - self.frame_num - 1
                print(f'{Fore.YELLOW}WARNING: Missed {num_of_missed_frames} frame', end='')
                if abs(num_of_missed_frames) > 1:
                    print('s.')
                else:
                    print('.')
        self.frame_num = header['frame_num']

    def len_check(self, received, expected, echo=False):
        if received is None or received < expected:
            if echo:
                print(f'{Fore.RED}ERROR: Corrupted frame.')
            return False
        return True

    def convert_idx(self, x, qformat=0):
        if x > 32767:
            x -= 65536

        return x/2**qformat

    def convert_detected_points(self, points):
        points['descriptor']['converted'] = True
        qformat = points['descriptor']['xyz_q_format']
        for i, obj in enumerate(points['objs']):
            doppler_idx = self.convert_idx(obj['doppler_idx'])
            doppler = doppler_idx * self.formats.doppler_resolution_mps

            range_val = obj['range_idx'] * self.formats.range_idx_to_meters

            peak = 10*np.log10(1+obj['peak_value'])
            x = self.convert_idx(obj['x_coord'], qformat=qformat)
            y = self.convert_idx(obj['y_coord'], qformat=qformat)
            z = self.convert_idx(obj['z_coord'], qformat=qformat)

            points['objs'][i] = {
                'x': x, 'y': y, 'z': z, 'range': range_val,
                'doppler': doppler, 'peak': peak
            }

    @staticmethod
    def pprint(frame):
        if frame is None:
            print(f'{Fore.MAGENTA}Frame: {Fore.RED}None')
            return

        print(Fore.CYAN + '='*85)
        Parser.pprint_struct(frame)
        print(Fore.CYAN + '='*85)
        print()

    @staticmethod
    def pprint_struct(frame, indentation='', _recursive_call=False):
        identetion_marker = '|' + 3*' '

        if not _recursive_call:
            Parser.pprint_struct.num = 1

        for key, value in frame.items():
            if isinstance(value, dict):
                print(indentation, end='')
                color = Parser.pprint_get_color(Parser.pprint_struct.num - 1)
                print(f'{color}{key}:')

                indentation += identetion_marker
                Parser.pprint_struct.num += 1
                Parser.pprint_struct(value, indentation, True)
                Parser.pprint_struct.num -= 1
                indentation = indentation[:-4]

                if key != sorted(frame.keys())[-1]:
                    print(indentation)

            elif isinstance(value, list):
                print(indentation, end='')
                color = Parser.pprint_get_color(Parser.pprint_struct.num - 1)
                print(f'{color}{key}:')

                indentation += identetion_marker
                for obj in value:
                    Parser.pprint_struct.num += 1
                    Parser.pprint_struct(obj, indentation, True)
                    Parser.pprint_struct.num -= 1
                    if obj != value[-1]:
                        print(indentation)

                indentation = indentation[:-4]

            else:
                if isinstance(value, tuple):
                    print(indentation, end='')
                    color = Parser.pprint_get_color(Parser.pprint_struct.num - 1)
                    print(f'{color}{key}:')
                    indentation += identetion_marker

                    with pd.option_context('display.max_rows', 10,
                                           'display.max_columns', 6,
                                           'display.precision', 2,
                                           'display.show_dimensions', False):
                        df = pprint.pformat(pd.DataFrame(value))
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
                    color = Parser.pprint_get_color(Parser.pprint_struct.num - 1)
                    print(f'{color}{key}: {value}')

    @staticmethod
    def pprint_get_color(num_of_identations):
        if num_of_identations % 4 == 0:
            return Fore.BLUE
        elif num_of_identations % 3 == 0:
            return Fore.CYAN
        elif num_of_identations % 2 == 0:
            return Fore.GREEN
        else:
            return Fore.MAGENTA
