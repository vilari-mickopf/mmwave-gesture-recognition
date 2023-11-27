#!/usr/bin/env python

import re
import time
import termios

import serial
import serial.tools.list_ports

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Connection:
    TIMEOUT = .5
    BAUDRATES = [
        9600,
        38400,
        19200,
        57600,
        115200,
        921600
    ]

    def __init__(self, name, rate=None):
        self.name = name
        self._rate = rate
        self.port = None

    def connected(self):
        return self.port is not None

    def if_connected(fn):
        def wrapper(self, *args, **kwargs):
            if self.connected():
                return fn(self, *args, **kwargs)
        return wrapper

    @property
    def rate(self):
        return self._rate

    @rate.setter
    @if_connected
    def rate(self, rate):
        detected_rate = self.get_baudrate()

        if rate is None:
            if detected_rate is None:
                print(f'{Fore.YELLOW}Is the connection open?',
                      f'{Fore.YELLOW}If not, baudrate should be specified manually.')
                print(f'{Fore.RED}No valid baud rate detected.',
                      f'{Fore.RED}Please check your connection.')
            else:
                rate = detected_rate
                print(f'Detected baud rate {Fore.BLUE}{detected_rate}.')

        if rate is None or rate not in self.BAUDRATES:
            print(f'{Fore.RED}Invalid baud rate.')
            self._rate = None
            return

        if None not in [detected_rate, rate] and rate != detected_rate:
            print(f'{Fore.YELLOW}Specified rate {rate}, but detected {detected_rate}.')

        self._rate = rate
        self.port.baudrate = self._rate
        print(f'Selecting baud rate {Fore.BLUE}{self.rate}')

    def connection_error_handler(fn):
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except(serial.SerialException, OSError):
                print(f'{Fore.RED}Port \'{self.name}\' not available.',
                      f'{Fore.RED}Check your connection and permissions')
                self.print_available_ports()
                self.port = None
            except termios.error:
                pass

        return wrapper

    @connection_error_handler
    def connect(self):
        if self.connected():
            self.disconnect()

        print(f'Connecting to port {Fore.BLUE}\'{self.name}\'')
        self.port = serial.Serial(self.name,
                                  bytesize=serial.EIGHTBITS,
                                  parity=serial.PARITY_NONE,
                                  stopbits=serial.STOPBITS_ONE,
                                  xonxoff=False,
                                  rtscts=False,
                                  dsrdtr=False,
                                  writeTimeout=0,
                                  timeout=self.TIMEOUT)

        self.rate = self._rate
        if self.rate is None:
            self.port = None

    def reset(self):
        self.close()
        self.open()

    @if_connected
    @connection_error_handler
    def open(self):
        self.port.open()

    @if_connected
    @connection_error_handler
    def close(self):
        self.port.close()

    @if_connected
    @connection_error_handler
    def flush(self):
        self.port.send_break(.1)
        self.port.flushInput()
        self.port.flushOutput()

    def disconnect(self):
        print(f'Disconnecting port {Fore.BLUE}{self.name}')
        self.close()
        time.sleep(.5)
        self.port = None

    @if_connected
    @connection_error_handler
    def write(self, data):
        self.port.write(data)
        return self.port.readline()

    @if_connected
    @connection_error_handler
    def read(self, size=None):
        if size is None:
            size = self.port.in_waiting
        return self.port.read(size)

    @if_connected
    @connection_error_handler
    def readline(self):
        return self.port.readline()

    def check_baudrate(self, rate):
        test_packet = 'qWeRtYuIoPaSdFgHjKLzXcVbNm\n'
        self.port.baudrate = rate
        self.port.write(test_packet.encode())

        time.sleep(.01)
        return test_packet.encode() in self.read()

    def get_baudrate(self):
        for baudrate in reversed(Connection.BAUDRATES):
            if self.check_baudrate(baudrate):
                return baudrate

    @staticmethod
    def print_available_ports():
        print(f'{Fore.YELLOW}Available ports:')
        for available_port in serial.tools.list_ports.comports():
            print(f'\t{Fore.YELLOW}{available_port}')


class mmWave:
    def __init__(self, cli_port, data_port=None, cli_rate=None, data_rate=None):
        self.cli_port = Connection(cli_port, cli_rate)
        if data_port is None or data_port == cli_port:
            self.data_port = self.cli_port
        else:
            self.data_port = Connection(data_port, data_rate)

        self.config_file = None

    @classmethod
    def from_config(cls, config):
        data_port, data_rate = config.get(['data_portname']), config.get(['data_baudrate'])
        cli_port, cli_rate = config.get(['cli_portname']), config.get(['cli_baudrate'])

        return cls(data_port=data_port, cli_port=cli_port,
                   data_rate=data_rate, cli_rate=cli_rate)

    def check_console(self):
        self.send_cmd('\n')
        response = self.get_cmd()
        if response is None:
            return False
        elif response == '':
            print(f'{Fore.YELLOW}mmWave cli no init response. Flash mode is active.')
            return False
        elif 'mmwDemo:/>' not in response:
            print(f'{Fore.RED}mmWave cli wrong init response: {response}.')
            return False
        return True

    def configure(self, config_file):
        with open(config_file, 'r') as f:
            lines = f.readlines()

        print('Configuring mmWave:')
        for line in lines:
            # Skip comment lines and blank lines
            if re.match(r'(^\s*%|^\s*$)', line):
                continue

            if not self.check_console():
                self.config_file = None
                return False

            # Send cmd
            print(f'Sending:  {Fore.YELLOW}{line}', end='')
            response = self.send_cmd(line)
            if response is None:
                self.config_file = None
                return False

            # Parse response
            response = self.get_cmd()
            response = response.replace('mmwDemo:/>', '')
            if 'Done' in response:
                print(f'Received: {Fore.GREEN}{response}')
            elif 'Ignored' in response or 'Debug:' in response:
                print(f'Received: {Fore.YELLOW}{response}')
            else:
                print(f'Received: {Fore.RED}{response}')
                print(f'{Fore.RED}Failed sending configuration')
                self.reset()
                self.config_file = None
                return False

            if 'sensorStart' not in line:
                time.sleep(0.01)

        self.config_file = config_file
        return True

    def connect(self):
        self.cli_port.connect()
        if self.data_port is not self.cli_port:
            self.data_port.connect()

        time.sleep(.5)
        self.get_cmd()
        if self.connected():
            self.check_console()

    def disconnect(self):
        self.cli_port.disconnect()
        if self.data_port is not self.cli_port:
            self.data_port.disconnect()

    def reset(self):
        self.data_port.reset()
        if self.cli_port is not self.data_port:
            self.cli_port.reset()

    def configured(self):
        return self.config_file is not None

    def connected(self):
        return self.cli_port.connected() and self.data_port.connected()

    @staticmethod
    def find_ports(pattern='XDS110'):
        ports = [str(p).split()[0] for p in serial.tools.list_ports.comports()
                 if pattern in str(p)]
        ports.sort()
        return ports

    def send_cmd(self, cmd):
        return self.cli_port.write(cmd.encode())

    def get_cmd(self):
        # Read the first byte
        first_byte = self.cli_port.read(1)

        # If the first byte is received, set the timeout to 0.05 seconds
        if first_byte:
            self.cli_port.port.timeout = 0.05

        response = first_byte + b''.join(iter(lambda: self.cli_port.read(1), b''))
        if not response:
            return None

        self.cli_port.port.timeout = self.cli_port.TIMEOUT
        return response.decode(errors='ignore')

    def get_data(self, size=None):
        return self.data_port.read(size=size)
