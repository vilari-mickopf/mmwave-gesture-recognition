#!/usr/bin/env python

import os
import re
from copy import deepcopy

from enum import Enum, EnumMeta, auto

import numpy as np


class Formats:
    MAGIC_NUMBER = b'\x02\x01\x04\x03\x06\x05\x08\x07'

    PROFILE_CFG_FORMAT = {
        'sensorStop': None,
        'flushCfg': None,
        'dfeDataOutputMode': {'modeType': int},
        'channelCfg': {
            'rxAntBitmap': int,
            'txAntBitmap': int,
            'cascading': int
        },
        'adcCfg': {
            'numADCBits': int,
            'adcOutputFmt': int
        },
        'adcbufCfg': {
            'subFrameIdx': int,
            'adcOutputFmt': int,
            'SampleSwap': int,
            'chanInterleave': int,
            'chirpThreshold': int
        },
        'profileCfg': {
            'profileId': int,
            'startFreq': float,
            'idleTime': float,
            'adcStartTime': float,
            'rampEndTime': float,
            'txOutPower': int,
            'txPhaseShifter': int,
            'freqSlopeConst': float,
            'txStartTime': float,
            'numAdcSamples': int,
            'digOutSampleRate': int,
            'hpfCornerFreq1': int,
            'hpfCornerFreq2': int,
            'rxGain': int
        },
        'chirpCfg': {
            'chirpStartIdx': int,
            'chirpEndIdx': int,
            'profileId': int,
            'startFreqVar': float,
            'freqSlopeVar': float,
            'idleTimeVar': float,
            'adcStartTimeVar': float,
            'txAntBitmask': int
        },
        'frameCfg': {
            'chirpStartIdx': int,
            'chirpEndIdx': int,
            'nLoops': int,
            'nFrames': int,
            'framePeriod': float,
            'triggerSelect': int,
            'triggerDelay': float
        },
        'lowPower': {
            'subFrameIdx': int,
            'adcMode': int
        },
        'guiMonitor': {
            'subFrameIdx': int,
            'detectedObjects': int,
            'logMagnitudeRange': int,
            'noiseProfile': int,
            'rangeAzimuthHeatMap': int,
            'rangeDopplerHeatMap': int,
            'statsInfo': int
        },
        'cfarCfg': {
            'subFrameIdx': int,
            'procDirection': int,
            'averageMode': int,
            'winLen': int,
            'guardLen': int,
            'noiseDiv': int,
            'cyclicMode': int,
            'thresholdScale': int
        },
        'peakGrouping': {
            'subFrameIdx': int,
            'groupingMode': int,
            'rangeDimEn': int,
            'dopplerDimEn': int,
            'startRangeIdx': int,
            'endRangeIdx': int
        },
        'multiObjBeamForming': {
            'subFrameIdx': int,
            'enabled': int,
            'threshold': float
        },
        'clutterRemoval': {
            'subFrameIdx': int,
            'enabled': int
        },
        'calibDcRangeSig': {
            'subFrameIdx': int,
            'enabled': int,
            'negativeBinIdx': int,
            'positiveBinIdx': int,
            'numAvgFrames': int
        },
        'extendedMaxVelocity': {
            'subFrameIdx': int,
            'enabled': int
        },
        'bpmCfg': {
            'subFrameIdx': int,
            'isEnabled': int,
            'chirp0Idx': int,
            'chirp1Idx': int
        },
        'lvdsStreamCfg': {
            'subFrameIdx': int,
            'isHeaderEnabled ': int,
            'dataFmt': int,
            'isSwEnabled': int
        },
        'nearFieldCfg': {
            'subFrameIdx': int,
            'enabled': int,
            'startRangeIndex': int,
            'endRangeIndex': int
        },
        'compRangeBiasAndRxChanPhase': {
            'rangeBias': float,
            'Re00': float,
            'Im00': float,
            'Re01': float,
            'Im01': float,
            'Re02': float,
            'Im02': float,
            'Re03': float,
            'Im03': float,
            'Re10': float,
            'Im10': float,
            'Re11': float,
            'Im11': float,
            'Re12': float,
            'Im12': float,
            'Re13': float,
            'Im13': float
        },
        'measureRangeBiasAndRxChanPhase': {
            'enabled': int,
            'targetDistance': float,
            'searchWin': float
        },
        'CQRxSatMonitor': {
            'profileIndx': int,
            'satMonSel': int,
            'primarySliceDuration': int,
            'numSlices': int,
            'rxChannelMask': int
        },
        'CQSigImgMonitor': {
            'profileIndx': int,
            'numSlices': int,
            'timeSliceNumSamples': int
        },
        'analogMonitor': {
            'rxSatMonEn ': int,
            'sigImgMonEn ': int
        }
    }

    HEADER_FORMAT = {
        'magic': '8s',
        'version': '4s',
        'packet_len': 'I',
        'platform': '4s',
        'frame_num': 'I',
        'time_cpu_cyc': 'I',
        'num_det_obj': 'I',
        'num_tlvs': 'I',
        'unknown': 'I'
    }

    TLVS_FORMAT = {
        'detectedPoints': {
            'descriptor': {
                'num_det_obj': 'H',
                'xyz_q_format': 'H'
            },
            'objs': [
                'descriptor.num_det_obj',
                 {
                    'range_idx': 'H',
                    'doppler_idx': 'H',
                    'peak_value': 'H',
                    'x_coord': 'H',
                    'y_coord': 'H',
                    'z_coord': 'H'
                }
            ]
        },
        'rangeProfile': '%sf',
        'noiseProfile': '%sf',
        'rangeAzimuthHeatMap': '%sf',
        'rangeDopplerHeatMap': '%sf',
        'stats': {
            'Inter-frame Processing Time': 'I',
            'Transmit Output Time': 'I',
            'Inter-frame Processing Margin': 'I',
            'Inter-chirp Processing Margin': 'I',
            'Active Frame CPU Load': 'I',
            'InterframeCPU Load': 'I'
        }
    }

    def __init__(self, config):
        self.config = self.parse(config)
        self.get_params()

        self.header = self.config_header()
        self.tlvs = self.config_tlvs()

    def get_antenna_num(self, bitmap):
        num = 0
        for i in range(int(np.floor(np.log2(bitmap)) + 1)):
            num += (bitmap >> i) & 1
        return num

    def get_params(self):
        channel_cfg = self.config['channelCfg']
        self.rx = self.get_antenna_num(channel_cfg['rxAntBitmap'])
        self.tx = self.get_antenna_num(channel_cfg['txAntBitmap'])

        frame_cfg = self.config['frameCfg']
        self.chirps_per_frame = frame_cfg['chirpEndIdx']
        self.chirps_per_frame -= frame_cfg['chirpStartIdx'] - 1
        self.chirps_per_frame *= frame_cfg['nLoops']

        profile_cfg = self.config['profileCfg']
        self.range_bins = int(2**np.ceil(np.log2(profile_cfg['numAdcSamples'])))
        self.doppler_bins = self.chirps_per_frame//self.tx

        range_factor = 300 * profile_cfg['digOutSampleRate']
        range_factor /= 2 * profile_cfg['freqSlopeConst'] * 1e3
        self.range_resolution_meters = range_factor/profile_cfg['numAdcSamples']
        self.range_idx_to_meters = range_factor/self.range_bins

        ramp_time = profile_cfg['idleTime'] + profile_cfg['rampEndTime']

        self.doppler_resolution_mps = 3e8
        self.doppler_resolution_mps /= 2 * profile_cfg['startFreq'] * 1e9
        self.doppler_resolution_mps /= 1e-6 * self.chirps_per_frame
        self.doppler_resolution_mps /= ramp_time

        self.max_range = 300 * 0.8 * profile_cfg['digOutSampleRate']
        self.max_range /= 2 * profile_cfg['freqSlopeConst'] * 1e3

        self.max_velocity = 3e8
        self.max_velocity /= 4 * profile_cfg['startFreq'] * 1e9 * 1e-6
        self.max_velocity /= self.tx * ramp_time

    def parse(self, config_file):
        format = self.PROFILE_CFG_FORMAT

        with open(config_file, 'r') as f:
            lines = f.readlines()

        # Initialize config
        config = {}
        for line in lines:
            line = line.strip()

            # Skip comment lines and blank lines
            if re.match(r'(^\s*%|^\s*$)', line):
                continue

            words = line.split()
            cmd = words[0]
            if format.get(cmd) is not None:
                config[cmd] = {}
                for i, param in enumerate(format[cmd]):
                    config[cmd][param] = format[cmd][param](words[i+1])

        return config

    def config_header(self):
        return deepcopy(self.HEADER_FORMAT)

    def config_tlvs(self):
        format = deepcopy(self.TLVS_FORMAT)

        adc_samples = self.config['profileCfg']['numAdcSamples']
        range_bins = self.range_bins
        if range_bins - adc_samples >= adc_samples - range_bins/2:
            range_bins //= 2

        # Size: RangeBins
        format['rangeProfile'] %= str(range_bins)

        # Size: RangeBins
        format['noiseProfile'] %= str(range_bins)

        # Size: RangeBins * num_virtual_antennas
        format['rangeAzimuthHeatMap'] %= str(range_bins * self.rx * self.tx)

        # Size: RangeBins * DopplerBins
        frame_cfg = self.config['frameCfg']
        chirps_per_frame = frame_cfg['chirpEndIdx'] - frame_cfg['chirpStartIdx']
        format['rangeDopplerHeatMap'] %= str(range_bins * chirps_per_frame//self.tx)

        return format


class GESTURE_META(EnumMeta):
    def __getitem__(self, index_or_name):
        if isinstance(index_or_name, str):
            return super().__getitem__(index_or_name.upper())

        index_or_name = int(index_or_name)
        if index_or_name < super().__len__():
            return list(self)[index_or_name]

    def __contains__(cls, index_or_name):
        if isinstance(index_or_name, cls):
            return True

        if isinstance(index_or_name, str):
            return index_or_name.upper() in cls.__members__.keys()

        index_or_name = int(index_or_name)
        return index_or_name in [v.value for v in cls.__members__.values()]


class GESTURE(Enum, metaclass=GESTURE_META):
    NONE = 0
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    CW = auto()
    CCW = auto()
    # Z = auto()
    # S = auto()
    # X = auto()

    @property
    def dir(self):
        dir = getattr(self, '_dir', os.path.join(os.path.dirname(__file__)))
        return os.path.join(dir, self.name.lower())

    @dir.setter
    def dir(self, path):
        self._dir = path

    def last_file(self):
        if not os.path.exists(self.dir) or not os.listdir(self.dir):
            return

        nums = []
        for f in os.listdir(self.dir):
            num = os.path.splitext(f)[0].split('_')[1]
            nums.append(int(num))
        last_sample = f'sample_{str(max(nums))}.npz'
        return os.path.join(self.dir, last_sample)

    def next_file(self):
        last_sample = self.last_file()
        if last_sample is None:
            return os.path.join(self.dir, 'sample_1.npz')

        last_sample_name = os.path.splitext(last_sample)[0]
        num = int(os.path.basename(last_sample_name).split('_')[1]) + 1
        return os.path.join(self.dir, f'sample_{str(num)}.npz')
