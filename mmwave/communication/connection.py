#! /usr/bin/env python

import re
import sys
import time

import serial
import serial.tools.list_ports

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Connection:
    TIMEOUT = 1
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
        self.rate = rate
        self.port = None

    def valid_baudrate(self, rate):
        if rate is not None and rate not in self.BAUDRATES:
            print(f'{Fore.RED}Baudate not valid.')
            return False
        return True

    def connected(self):
        if self.port is None:
            return False
        return True

    def if_connected(fn):
        def wrapper(self, *args, **kwargs):
            if self.connected():
                return fn(self, *args, **kwargs)
        return wrapper

    def connect(self):
        if self.connected():
            self.disconnect()

        if not self.valid_baudrate(self.rate):
            return False

        rate = self.rate
        if rate is None:
            rate = self.get_baudrate(self.name)

        if rate is None:
            return False

        print(f'Connecting to port {Fore.BLUE}\'{self.name}\'')
        try:
            self.port = serial.Serial(self.name,
                                      bytesize=serial.EIGHTBITS,
                                      parity=serial.PARITY_NONE,
                                      stopbits=serial.STOPBITS_ONE,
                                      xonxoff=False,
                                      rtscts=False,
                                      dsrdtr=False,
                                      writeTimeout=0,
                                      timeout=self.TIMEOUT)

        except(serial.SerialException, OSError):
            self.connection_error()
            return False

        print(f'Selecting baud rate {Fore.BLUE}{rate}')
        self.port.baudrate = rate
        return True

    @staticmethod
    def print_available_ports():
        print(f'{Fore.YELLOW}Available ports:')
        for available_port in serial.tools.list_ports.comports():
            print(f'\t{Fore.YELLOW}{available_port}')

    def connection_error(self):
        print(f'{Fore.RED}Port \'{self.name}\' not available.',
              f'{Fore.RED}Check your connection and permissions')
        self.print_available_ports()
        self.port = None

    def set_baudrate(self, rate=None):
        if not self.valid_baudrate(rate):
            return

        self.rate = rate
        if rate is None:
            rate = self.get_baudrate(self.name)

        if rate is None:
            return

        if self.connected():
            self.port.baudrate = rate

    def reset(self):
        self.close()
        self.open()

    @if_connected
    def open(self):
        self.port.open()

    @if_connected
    def close(self):
        self.port.close()

    @if_connected
    def flush(self):
        self.port.send_break(.1)
        self.port.flushInput()
        self.port.flushOutput()

    @if_connected
    def disconnect(self):
        print(f'Disconnecting port {Fore.BLUE}{self.name}')
        self.close()
        time.sleep(1)

        self.port = None

    @if_connected
    def write(self, data, encoding=None, size=None):
        try:
            if encoding is not None:
                self.port.write(bytes(data, encoding=encoding))
            else:
                self.port.write(bytes(data))

            if size is not None:
                response = self.port.read(size)
            else:
                self.port.readline()
                response = self.port.readline()
            return response
        except(serial.SerialException, OSError):
            self.connection_error()

    @if_connected
    def read(self, size=None):
        try:
            if size is not None:
                data = self.port.read(size)
            else:
                data = self.port.read(self.port.in_waiting)
            return data
        except(serial.SerialException, OSError):
            self.connection_error()

    @staticmethod
    def get_baudrate(port):
        print('Automatic baud rate search...', end='', flush=True)
        test_packet = 'qWeRtYuIoPaSdFgHjKLzXcVbNm\n'

        try:
            test_conn = serial.Serial(port, timeout=Connection.TIMEOUT,
                                            write_timeout=Connection.TIMEOUT)
        except(serial.SerialException, OSError):
            print(f'{Fore.RED}Can\'t write to port.',
                  'Please check your connection or permissions.')
            return None

        detected_baudrate = None
        for baudrate in reversed(Connection.BAUDRATES):
            test_conn.baudrate = baudrate

            start_time = time.time()
            while time.time() - start_time < Connection.TIMEOUT:
                try:
                    test_conn.write(bytes(test_packet, encoding='ascii'))
                except(serial.SerialTimeoutException, serial.SerialException, OSError):
                    print(f'{Fore.RED}Can\'t write to port.',
                          f'{Fore.RED}Please check your connection or permissions.')
                    return None

                time.sleep(.01)
                response = test_conn.read(test_conn.in_waiting)
                if bytes(test_packet, encoding='ascii') in response:
                    detected_baudrate = baudrate
                    break

            if detected_baudrate is not None:
                print(f'Detected baud rate {Fore.BLUE}{detected_baudrate}.')
                break
        else:
            print()
            print(f'{Fore.YELLOW}Is the connection open?',
                  f'{Fore.YELLOW}If not, baudrate should be specified manually.')
            print(f'{Fore.RED}No valid baud rate detected.',
                  f'{Fore.RED}Please check your connection.')

        test_conn.close()
        return detected_baudrate


class mmWave:
    def __init__(self, cli_port, data_port=None,
                       cli_rate=None, data_rate=None):
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

    def configure(self, config_file):
        with open(config_file, 'r') as f:
            lines = f.readlines()

        print('Configuring mmWave:')
        for line in lines:
            # Skip comment lines and blank lines
            if re.match(r'(^\s*%|^\s*$)', line):
                continue

            # Send cmd
            print(f'Sending:  {Fore.YELLOW}{line}', end='')
            response = self.cli_port.write(line, encoding='ascii')
            if response is None:
                self.config_file = None
                return False

            # Parse response
            response = response.decode('ascii', errors='ignore')
            if 'Done' not in response and 'sensorStart' not in line:
                print(f'Received: {Fore.RED}{response}')
                print(f'{Fore.RED}Failed sending configuration')
                self.reset()
                self.config_file = None
                return False
            else:
                print(f'Received: {Fore.GREEN}{response}')

            if 'sensorStart' not in line:
                time.sleep(0.01)

        self.config_file = config_file
        return True

    def connect(self):
        self.cli_port.connect()
        if self.data_port is not self.cli_port:
            self.data_port.connect()

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
        ports = []
        for port in serial.tools.list_ports.comports():
            if pattern in str(port):
                ports.append(str(port).split()[0])

        if sys.platform.system() == 'Windows':
            ports.sort(reverse=True)
        else:
            ports.sort()

        return ports

    def send_cmd(self, data, encoding=None, size=None):
        return self.cli_port.write(data, encoding=encoding, size=size)

    def get_cmd(self, size=None):
        return self.cli_port.read(size=size)

    def get_data(self, size=None):
        return self.data_port.read(size=size)
