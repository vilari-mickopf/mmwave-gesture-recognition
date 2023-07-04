#!/usr/bin/env python

import glob
import os
import platform
import readline
import binascii
import queue
import serial

from cmd import Cmd

from sklearn.model_selection import train_test_split

from mmwave.communication import Connection, mmWave, Parser
from mmwave.data import Formats, GESTURE, Logger
from mmwave.model import ConvModel, LstmModel, TransModel
from mmwave.utils import Plotter, print, error, warning
from mmwave.utils.flasher import Flasher, CMD, OPCODE

from handlers import SignalHandler, Completer
from threads import threaded, ListenThread, ParseThread, PrintThread
from threads import LogThread, PredictThread, PlotThread

import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Console(Cmd):
    def __init__(self, main_queue):
        super().__init__()

        # Flash
        self.firmware_dir = os.path.join(os.path.dirname(__file__), 'firmware')
        self.flasher = None

        # Connection
        self.default_cli_rate = 115200
        self.default_data_rate = 921600
        self.mmwave_init(cli_port=None, data_port=None,
                         cli_rate=self.default_cli_rate,
                         data_rate=self.default_data_rate)

        if self.mmwave is None or not self.mmwave.connected():
            print('Try connecting manually. Type \'help connect\' for more info.\n')

        # Configuration
        self.config_dir = os.path.join(os.path.dirname(__file__),
                                       'mmwave/communication/profiles')
        self._configured = None
        self.default_config = 'profile'
        self.config = None
        self.parser = None

        # Data logger
        self.logger = Logger()
        self.data_dir = None

        # Model
        self.model = 'lstm'

        # Threading stuff
        self.main_queue = main_queue
        self.plotter_queue = queue.Queue()
        self.plotter = Plotter(self.plotter_queue)

        self.listen_thread = None
        self.parse_thread = None
        self.print_thread = None
        self.plot_thread = None
        self.log_thread = None
        self.predict_thread = None

        # Catching signals (ctrl-c)
        self.console_queue = queue.Queue()
        SignalHandler(self.console_queue)

        self.set_prompt()
        print(f'{Fore.GREEN}Init done.\n')
        print(f'{Fore.MAGENTA}--- mmWave console ---')
        warning('Type \'help\' for more information.')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model == 'conv':
            self._model = ConvModel()
        elif model == 'lstm':
            self._model = LstmModel()
        elif model == 'trans':
            self._model = TransModel()
        else:
            raise NotImplemented(model)

    @property
    def configured(self):
        return self._configured

    @configured.setter
    def configured(self, value):
        self._configured = value
        self.parser = Parser(Formats(self.config)) if value else None

    def mmwave_init(self, cli_port, data_port, cli_rate, data_rate):
        self.mmwave = None
        if cli_port is None or data_port is None:
            print('Looking for ports...')
            ports = mmWave.find_ports()

            if len(ports) < 2:
                print(f'{Fore.RED}Ports not found!')
                print(f'{Fore.YELLOW}Auto-detection is only applicable for',
                      f'{Fore.YELLOW}eval boards with XDS110.')
                return

            if len(ports) > 2:
                print(f'{Fore.YELLOW}Multiple ports detected.',
                      f'{Fore.YELLOW}Selecting ports {ports[0]} and {ports[1]}.')
                ports = ports[:2]

            if platform.system() == 'Windows':
                ports.sort(reverse=True)

        self.mmwave = mmWave(ports[0], ports[1],
                             cli_rate=cli_rate,
                             data_rate=data_rate)
        self.mmwave.connect()
        if self.mmwave.connected():
            self.flasher = Flasher(self.mmwave.cli_port)

    def connected(self):
        return self.mmwave is not None and self.mmwave.connected()

    def set_prompt(self):
        self.prompt = f'{Fore.RED}[Not connected]{Fore.RESET} >> '
        if self.connected():
            self.prompt = f'{Fore.GREEN}>>{Fore.RESET} '

    def preloop(self):
        '''
        Initialization before prompting user for commands.
        Despite the claims in the Cmd documentation, Cmd.preloop() is not a
        stub.
        '''
        Cmd.preloop(self)   # sets up command completion
        self._hist = []      # No history yet
        self._locals = {}      # Initialize execution namespace for user
        self._globals = {}

    def postloop(self):
        '''
        Take care of any unfinished business.
        Despite the claims in the Cmd documentation, Cmd.postloop() is not a
        stub.
        '''
        Cmd.postloop(self)   # Clean up command completion
        print('Exiting...')

    def check_thread(self, name):
        thread = getattr(self, f'{name}_thread')
        return thread is not None and thread.is_alive()

    def if_thread_running(name):
        def arg_wrapper(func):
            def func_wrapper(self, *args, **kwargs):
                if not self.check_thread(name):
                    error(f'{name} thread not started.')
                    return
                return func(self, *args, **kwargs)
            return func_wrapper
        return arg_wrapper

    def update_plotter(self):
        try:
            info = self.plotter_queue.get(False)
        except queue.Empty:
            return

        if info == 'closed':
            print(f'{Fore.YELLOW}Plotter closed.\n')
            if self.check_thread('plot'):
                self.plot_thread.stop()

    def precmd(self, line):
        '''
        This method is called after the line has been input but before
        it has been interpreted. If you want to modify the input line
        before execution (for example, variable substitution) do it here.
        '''
        self._hist += [line.strip()]
        self.update_plotter()

        return line

    def postcmd(self, stop, line):
        '''
        If you want to stop the console, return something that evaluates to
        true. If you want to do some post command processing, do it here.
        '''
        self.set_prompt()
        return stop

    def emptyline(self):
        '''Do nothing on empty input line'''
        pass

    def default(self, line):
        '''
        Called on an input line when the command prefix is not recognized.
        In that case we execute the line as Python code.
        '''
        try:
            exec(line) in self._locals, self._globals
        except Exception:
            error('Unknown arguments.')
            return

    def do_help(self, args):
        '''
        Get help on command

        \'help\' or \'?\' with no arguments prints a list of commands for
        which help is available \'help <command>\' or \'? <command>\' gives
        help on <command>
        '''
        # The only reason to define this method is for the help text in
        # the doc string
        Cmd.do_help(self, args)

    def do_history(self, args):
        '''Print a list of commands that have been entered'''
        print(self._hist)

    def complete_from_list(self, complete_list, text, line):
        mline = line.partition(' ')[2]
        offs = len(mline) - len(text)
        return [s[offs:] for s in complete_list if s.startswith(mline)]

    def if_connected(fn):
        def wrapper(self, *args, **kwargs):
            if self.connected():
                return fn(self, *args, **kwargs)
            error('Not connected.')
        return wrapper

    def argcheck(min=0, max=0):
        def arg_wrapper(func):
            def func_wrapper(self, *args, **kwargs):
                assert min <= max

                if not args:
                    args = ('', )

                if args[0] != '' and min == 0 and max == 0:
                    error('Unknown arguments.')
                    return

                if args[0] == '' and min != 0:
                    error('Too few arguments.')
                    return

                if len(args[0].split()) > max:
                    error('Too many arguments.')
                    return

                return func(self, *args, **kwargs)
            return func_wrapper
        return arg_wrapper

    @argcheck()
    def do_exit(self, args):
        '''Exits from the console'''
        self.do_stop()
        if self.connected():
            self.mmwave.disconnect()
        os._exit(0)

    @if_connected
    @argcheck(min=1, max=4)
    def do_flash(self, args=''):
        '''
        Sending .bin files to mmWave

        Flashing bin files to connected mmWave. It is possible to specify up to
        4 meta files. SOP0 and SOP2 have be closed and power reseted before
        running this command.

        Look \'connect\' and \'autoconnect\' command for connecting to mmWave.

        Usage:
        >> flash xwr16xx_mmw_demo.bin
        '''

        filepaths = []
        for arg in args.split():
            filepath = os.path.join(self.firmware_dir, arg)
            if not os.path.isfile(filepath):
                error(f'File \'{filepath}\' doesn\'t exist.')
                return
            filepaths.append(filepath)

        print('Ping mmWave...', end='')
        response = self.flasher.send_cmd(CMD(OPCODE.PING), resp=False)
        if response is None:
            warning('Check if SOP0 and SOP2 are closed, and reset the power.')
            return
        print(f'{Fore.GREEN}Done.')

        print('Get version...', end='')
        response = self.flasher.send_cmd(CMD(OPCODE.GET_VERSION))
        if response is None:
            return
        print(f'{Fore.GREEN}Done.')
        print(f'{Fore.BLUE}Version:', binascii.hexlify(response))
        print()

        self.flasher.flash(filepaths, erase=True)
        print(f'{Fore.GREEN}Done.')

    def complete_flash(self, text, line, begidx, endidx):
        bins = os.path.join(self.firmware_dir, '*.bin')
        completions = [os.path.basename(f) for f in glob.glob(bins)]

        return self.complete_from_list(completions, text, line)

    @argcheck()
    def do_autoconnect(self, args=''):
        '''
        Auto connecting to mmWave

        Automatically looking for \'XDS\' ports and connecting with default
        baudrates. Auto-detection is only applicable for eval boards with
        XDS110.

        Look \'connect\' command for manual connection.

        Usage:
        >> autoconnect
        '''
        if self.connected():
            warning('Already connected.')
            print('Reconnecting...')

        self.mmwave_init(cli_port=None, data_port=None,
                         cli_rate=self.default_cli_rate,
                         data_rate=self.default_data_rate)

    def get_user_port(self, type):
        old_completer = readline.get_completer()

        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(port[0])

        compl_ports = Completer(ports)
        readline.set_completer_delims('\t')
        readline.parse_and_bind('tab: complete')
        readline.set_completer(compl_ports.list_completer)

        port = None
        while port is None:
            port = input(f'{Fore.YELLOW}{type} port:{Fore.RESET} ').strip()
            if port in ['q', 'exit']:
                return
            elif port not in ports:
                error(f'Port {port} is not valid.')
                warning('Valid ports:')
                for port in ports:
                    warning(f'\t{port}')
                warning('Type \'exit\' to return.')
                port = None

        readline.set_completer(old_completer)
        return port

    def get_user_rate(self, type):
        old_completer = readline.get_completer()

        rates = [str(rate) for rate in Connection.BAUDRATES]
        compl_rates = Completer(rates)
        readline.set_completer(compl_rates.list_completer)
        rate = -1
        while rate == -1:
            rate = input(f'{Fore.YELLOW}{type} rate:{Fore.RESET} ').strip()
            if rate in ['q', 'exit']:
                break
            elif rate == '':
                rate = None
            elif rate not in rates:
                error(f'Rate {rate} is not valid.')
                warning('Valid baudrates:')
                for rate in rates:
                    warning('\t' + rate)
                warning('Type \'exit\' to return.')
                rate = -1
            else:
                rate = int(rate)

        readline.set_completer(old_completer)
        return rate

    @argcheck()
    def do_connect(self, args=''):
        '''
        Manually connecting to mmWave

        Command will ask you to manualy type ports and baudrates.
        Use <Tab> for autocompletion on available ports and baudrates.
        Type \'exit\' for exiting manual connection.
        If you only press enter on baudrates, console will try to find
        baudrate automatically. Note that this will work only if connection
        is already opened from both sides. (On mmWave startup only cli port
        is open, and data port will open only after \'senstrStart\' command)

        Look \'autoconnect\' command for automatic connection.

        Usage:
        >> connect
        cli port: /dev/ttyACM0
        cli rate: 115200
        data port: /dev/ttyACM1
        data rate: 921600
        '''

        if self.connected():
            self.mmwave.disconnect()
            print()

        ports = {}
        for name in ['cli', 'data']:
            port = self.get_user_port(name)
            if port is not None:
                ports[f'{name}_port'] = port
                ports[f'{name}_rate'] = self.get_user_rate(name)
            else:
                return

        self.mmwave_init(**ports)

    @if_connected
    @argcheck(max=1)
    def do_configure(self, args=''):
        '''
        Sending configuration to mmWave

        Configuring mmWave with given configuration file. All configuration
        files should be placed in \'mmwave/communication/profiles\' folder.
        Use <Tab> for autocompletion on available configuration files.
        All configuration files should have .cfg extension.
        If no configuration is provided, default configuration file will be
        used.

        Usage:
        >> configure
        >> configure profile
        '''
        if args == '':
            args = self.default_config

        config = os.path.join(self.config_dir, f'{args}.cfg')
        if config not in glob.glob(os.path.join(self.config_dir, '*.cfg')):
            error('Unknown profile.')
            return

        if not self.configured and os.path.basename(config) == 'start.cfg':
            error('Can\'t start. mmWave not configured.')
            return

        mmwave_configured = self.mmwave.configure(config)
        if not mmwave_configured:
            self.configured = False
            return

        if os.path.basename(config) == 'start.cfg':
            return
        elif os.path.basename(config) == 'stop.cfg':
            self.configured = False
            if self.check_thread('listen'):
                warning('Listen thread alredy started. Stopping...')
                self.listen_thread.stop()

        self.config = config
        self.configured = False if os.path.basename(config) == 'stop.cfg' else True

    def complete_configure(self, text, line, begidx, endidx):
        completions = []
        for file in glob.glob(os.path.join(self.config_dir, '*.cfg')):
            completions.append('.'.join(os.path.basename(file).split('.')[:-1]))

        return self.complete_from_list(completions, text, line)

    @argcheck(min=1, max=1)
    def do_set_model(self, args=''):
        '''
        Set model type used for prediction. Available models are
        \'conv\' (convolutional 1D), \'lstm\' (long short-term memory) and
        \'trans\' (transformer). Default is lstm.

        Usage:
        >> set_model conv
        >> set_model lstm
        >> set_model trans
        '''

        if args not in ['conv', 'lstm', 'trans']:
            warning(f'Unknown argument: {args}')
            return

        self.model = args

    def complete_set_model(self, text, line, begidx, endidx):
        return self.complete_from_list(['conv', 'lstm', 'trans'], text, line)

    @argcheck()
    def do_get_model(self, args=''):
        '''
        Get current model type.

        Usage:
        >> get_model
        '''

        print(f'Current model type: {self.model}')

    @argcheck()
    def do_listen(self, args=''):
        '''
        Start listener and parser thread

        Starting listener on connected ports. mmWave should be configured
        before using this option.

        Look 'connect' and 'autoconnect' command for connecting to mmWave.
        Look 'configure' command for connecting to mmWave.

        Usage:
        >> listen
        '''
        if not self.configured:
            warning('mmWave not configured.')
            return

        if self.check_thread('listen'):
            warning('listen thread already started.')
            return

        print(f'{Fore.CYAN}=== Listening ===')
        self.listen_thread = ListenThread(self.mmwave)
        self.parse_thread = ParseThread(self.parser)
        self.parse_thread.start()
        self.listen_thread.start()
        self.listen_thread.forward_to(self.parse_thread)

    def do_stop(self, args=''):
        '''
        Stopping mmwave, listener and plotter.

        Possible options: \'mmwave\', \'listen\' and \'plot\'.
            \'mmwave\': Sending \'sensorStop\' command to mmWave. This option
                        will also stop listener and plotter.
            \'listen\': Stopping listener and parser threads. This option
                        will also stop plotter.
            \'plot\': Closing plotter.
         If nothing is provided as a argument, \'mmwave\' will be used.

        Usage:
        >> stop
        >> stop plot
        '''
        if args == '' or 'mmwave' in args.split():
            opts = ['plot', 'listen', 'mmwave']
        elif 'listen' in args.split():
            opts = ['plot', 'listen']
        else:
            opts = args.split()

        if 'plot' in opts:
            if self.check_thread('plot'):
                self.plot_thread.stop()
                print('Plotter stopped.')
                self.plot_thread.send_to_main(self.plotter.close)
            opts.remove('plot')

        if 'listen' in opts:
            if self.check_thread('listen'):
                self.listen_thread.stop()
                print('Listener stopped.')
            opts.remove('listen')

        if 'mmwave' in opts:
            if self.configured:
                self.do_configure('stop')
                print('mmWave stopped.')
            opts.remove('mmwave')

        for opt in opts:
            warning(f'Unknown option: {opt}. Skipped.')

    def complete_stop(self, text, line, begidx, endidx):
        return self.complete_from_list(['mmwave', 'listen', 'plot'], text, line)

    @argcheck()
    @if_thread_running('listen')
    def do_print(self, args=''):
        '''
        Pretty print

        Printing received frames. Listener should be started before using
        this command. Use <Ctrl-C> to stop this command.

        Usage:
        >> print
        '''

        self.console_queue.queue.clear()
        self.print_thread = PrintThread(self.parser)
        self.print_thread.start()
        self.parse_thread.forward_to(self.print_thread)

        # Wait for user termination
        self.console_queue.get()
        self.print_thread.stop()

    @argcheck()
    @if_thread_running('listen')
    def do_plot(self, args=''):
        '''
        Start plotter

        Plotting received frames. Listener should be started before using
        this command. Use \'stop plot\' to stop this command.

        Usage:
        >> plot
        '''

        if self.check_thread('plot'):
            warning('Plot thread already started.')
            return

        self.plot_thread = PlotThread(self.plotter, self.main_queue)
        self.plot_thread.start()
        self.parse_thread.forward_to(self.plot_thread)

    @argcheck()
    @if_thread_running('listen')
    def do_predict(self, args=''):
        '''
        Start prediction

        Passing received frames through neural network and printing results.
        Listener should be started before using this command.
        Use <Ctrl-C> to stop this command.

        Usage:
        >> predict
        '''

        if not self.model.loaded():
            self.model.load()
            print()

        self.predict_thread = PredictThread(self.model)
        self.predict_thread.start()
        self.parse_thread.forward_to(self.predict_thread)

        # Wait for user termination
        self.console_queue.get()
        self.predict_thread.stop()

    @argcheck()
    def do_start(self, args=''):
        '''
        Start listener, plotter and prediction.

        If mmWave is not configured, default configuration will be send
        first. Use <Ctrl-C> to stop this command.

        Usage:
        >> start
        '''

        if not self.model.loaded():
            self.model.load()
            print()

        self.do_configure()
        self.do_listen()
        self.do_plot()
        self.do_predict()

    @argcheck(max=1)
    def do_train(self, args=''):
        '''
        Train neural network with data

        Usage:
        >> train
        >> train <DATA_PATH>
        '''

        # TODO: ability to set data dir

        X, y = Logger.get_all_data()
        Logger.get_stats(X, y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3,
                                                          stratify=y,
                                                          random_state=12)
        self.model.train(X_train, y_train, X_val, y_val)

    @argcheck(max=1)
    def do_eval(self, args=''):
        '''
        Evaluate neural network

        Usage:
        >> eval
        >> eval <DATA_PATH>
        '''

        # TODO: ability to set data dir

        X, y = Logger.get_all_data()
        Logger.get_stats(X, y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3,
                                                          stratify=y,
                                                          random_state=12)

        if not self.model.loaded():
            self.model.load()
            print()

        print('Eval validation dataset:')
        self.model.evaluate(X_val, y_val)
        print()

        print('Eval train dataset:')
        self.model.evaluate(X_train, y_train)
        print()

        print('Eval full dataset:')
        self.model.evaluate(X, y)
        print()

    def complete_eval(self, text, line, begidx, endidx):
        return self.complete_from_list(['refresh'], text, line)

    @argcheck(min=1, max=1)
    @if_thread_running('listen')
    def do_log(self, args=''):
        '''
        Log data

        Logging specified gesture. Data will be saved in
        \'mmwave/data/gesture_foder/sameple_num.csv\' file.
        Possible options: \'up\', \'down\', \'left\', \'right\', \'cw\',
                          \'ccw\', \'s\', \'z\', \'x\'

        Usage:
        >> log up
        >> log ccw
        >> log z
        '''

        if not args in GESTURE:
            warning(f'Unknown argument: {args}')
            return

        self.log_thread = LogThread(self.logger, GESTURE[args])
        self.log_thread.start()
        self.parse_thread.forward_to(self.log_thread)

    def complete_gestures(self, text, line):
        completions = []
        for gesture in GESTURE:
            completions.append(gesture.name.lower())
        return self.complete_from_list(completions, text, line)

    def complete_log(self, text, line, begidx, endidx):
        return self.complete_gestures(text, line)

    @argcheck(min=1, max=1)
    def do_remove(self, args=''):
        '''
        Remove last sample

        Removing last gesture sample. Data will be removed from
        \'mmwave/data/gesture_foder\' folder.
        Possible options: \'up\', \'down\', \'left\', \'right\', \'cw\',
                          \'ccw\', \'s\', \'z\', \'x\'
        Usage:
        >> remove up
        >> remove ccw
        >> remove z
        '''

        if not args in GESTURE:
            error(f'Unknown gesture: {args}')
            return

        self.logger.discard_last_sample(args)

    def complete_remove(self, text, line, begidx, endidx):
        return self.complete_gestures(text, line)

    @argcheck(min=1, max=1)
    @if_thread_running('plot')
    def do_redraw(self, args=''):
        '''
        Redraw sample

        Redrawing last captured gesture file.
        Possible options: \'up\', \'down\', \'left\', \'right\', \'cw\',
                          \'ccw\', \'s\', \'z\', \'x\'

        Usage:
        >> redraw up
        >> redraw ccw
        >> redraw z
        '''

        if args not in GESTURE:
            error(f'Unknown gesture: {args}')
            return

        self.plot_thread.send_to_main(self.plotter.draw_last_sample, GESTURE[args])

    def complete_redraw(self, text, line, begidx, endidx):
        return self.complete_gestures(text, line)


@threaded
def console_thread(console):
    while True:
        console.cmdloop()


def main(q):
    while True:
        func, args, kwargs = q.get()
        func(*args, **kwargs)


if __name__ == '__main__':
    main_queue = queue.Queue()
    console_thread(Console(main_queue))

    # All plotting stuff has to be done in main thread
    main(main_queue)
