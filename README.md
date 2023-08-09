# Basic Gesture Recognition Using mmWave Sensor - TI AWR1642

Collecting data from TI AWR1642 via serial port and passing it through convolutional, resnet, lstm or transformer neural network for recognizing one of following gestures:
- None (random non gestrures)
- Swipe Up
- Swipe Down
- Swipe Right
- Swipe Left
- Spin CW
- Spin CCW

![Demo](./demo.gif)

# Getting Started

## Deps:
- python 3.8+
- unzip (optional)
- curl (optional)

unzip and curl are used in `get-data` script, you can download [data]() and [models]() manually and put unzip contented them to [mmwave_gesture/data](./mmwave_gesture/data) and [mmwave_gesture](./mmwave_gesture) respectively.

## Installation

Install [mmwave_gesture](./mmwave_gesture/) package locally:

```bash
git clone https://gitlab.com/vilari-mickopf/mmwave-gesture-recognition.git
cd mmwave-gesture-recognition
pip install -e .
./get-data
```

## Serial permissions

The group name can differ from distribution to distribution.

### Arch

```bash
gpasswd -a <username> uucp
```

### Ubuntu:

```bash
gpasswd -a <username> dialout
```

The change will take effect on the next login.

The group name can be obtained by running:

```bash
stat /dev/ttyACM* | grep Gid
```

### One time only (permissions will be reseted after unplugging):

```bash
chmod 666 /dev/ttyACM*
```

## Flashing

The code used for AWR1642 is just a variation of mmWaveSDK demo provided with
the version 02.00.00.04. Bin file is located in [firmware](./firmware/) directory.

1. Close SOP0 and SOP2, and reset the power.
2. Start the console and run flash command:
```bash
python console.py
>> flash xwr16xx_mmw_demo.bin
```
3. Remove SOP0 and reset the power again.


## Running

If the board was connected before starting the console, the script should automatically find the ports and connect to them. This is only applicable for boards with XDS. If the board is connected after starting the console, _autoconnect_ command should be run. If for some reason this is not working, manual connection is available via _connect_ command. Manual connection can also be used for boards without XDS. Type _help connect_ or _help autoconnect_ for more info.

If the board is connected, the prompt will be green, otherwise, it will be red.

After connecting, simple _start_ command will start listener, parser, plotter and prediction.

```bash
python console.py
>> start
```

Use _Ctrl-C_ to stop this command.


### Collecting data

The console can be used for easy data collection. Use _log_ command to save gesture samples in .csv files in [mmwave/data/](./mmwave/data/) directory. If nothing is captured for more than a half a second, the command will automatically stop. _redraw_/_remove_ commands will redraw/remove the last captured sample.

```bash
python console.py
>> listen
>> plot
>> log up
>> log up
>> redraw up
>> remove up
>> log down
>> log ccw
```

### Training

```bash
python console.py
>> train
```

or

```bash
python mmwave_gesture/model.py
>> train
```

### Selecting model
By default, lstm model is used. Other models can be selected using _set_model_ option.
```bash
python console.py
>> set_model conv1d
>> set_model lstm
>> set_model trans
```

### Help

Use help command to list all available commands and get documentation on them.

```bash
python console.py
>> help
>> help flash
>> help listen
```

## Authors

* **Filip Markovic**

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Acknowledgments

* Thanks to [NOVELIC](https://www.novelic.com/) for providing me with sensors
