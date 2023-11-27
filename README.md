# Basic Gesture Recognition Using mmWave Sensor - TI AWR1642

Collecting data from the TI AWR1642 via its serial port, this setup allows the user to choose one of several neural network architectures - convolutional, ResNet, LSTM, or Transformer.
The selected network is then used for the recognition and classification of specific gestures:
- None (random non-gestures)
- Swipe Up
- Swipe Down
- Swipe Right
- Swipe Left
- Spin Clockwise
- Spin Counterclockwise
- Letter Z
- Letter S
- Letter X

[![Demo](https://i.imgur.com/QJKJhld.png)](https://drive.google.com/file/d/1CCb8JBcAVc_qRKH24a_IKks0S9raUnme/view?usp=sharing)


# Getting Started

## Deps:
- python 3.8+
- unzip (optional)
- curl (optional)

_unzip and curl are used by the [fetch](./fetch) script._


## Installation

Install [mmwave_gesture](./mmwave_gesture/) package locally:

```bash
git clone https://github.com/vilari-mickopf/mmwave-gesture-recognition.git
cd mmwave-gesture-recognition
pip install -e .
```

### Data and models
You can run [./fetch](./fetch) script to download and extract:

- [data](https://www.dropbox.com/scl/fi/y431rn0eauy2qkiz0y0g2/data.zip?rlkey=punhs9iquojldn6ug2owgnkbv&dl=0) (20k samples - 2k per class) ~120Mb

- [models](https://www.dropbox.com/scl/fi/ni8ioomcqzjvocfj9gx1j/models.zip?rlkey=pf0g7tpi20zn3idowptw9y9fe&dl=0) (Conv1D, Conv2D, ResNet1D, ResNet2D, LSTM and Transformer models) ~320Mb

To access the required data manually, follow the provided links to download the files.
Once downloaded, manually extract the contents to the directories [mmwave_gesture/data/](mmwave_gesture/data/) and [mmwave_gesture/models/](mmwave_gesture/models/) as appropriate.

End result should look like this:
```
mmwave_gesture/
│ communication/
│ data/
│ │ ccw/
│ │ cw/
│ │ down/
│ │ │ sample_1.npz
│ │ │ sample_2.npz
│ │ │ ...
│ │ └ sample_2000.npz
│ │ left/
│ │ none/
│ │ right/
│ │ s/
│ │ up/
│ │ x/
│ │ z/
│ │ __init__.py
│ │ formats.py
│ │ generator.py
│ │ loader.py
│ │ logger.py
│ └ preprocessor.py
│ models/
│ │ Conv1DModel/
│ │ │ confusion_matrix.png
│ │ │ history
│ │ │ model.h5
│ │ │ model.png
│ │ └ preprocessor
│ │ Conv2DModel/
│ │ LstmModel/
│ │ ResNet1DModel/
│ │ ResNet2DModel/
│ └ TransModel/
│ utils/
│ __init__.py
│ model.py
...
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
python mmwave-console.py
>> flash xwr16xx_mmw_demo.bin
```
3. Remove SOP0 and reset the power again.


## Running

If the board was connected before starting the console, the script should automatically find the ports and connect to them. This is only applicable for boards with XDS. If the board is connected after starting the console, _autoconnect_ command should be run. If for some reason this is not working, manual connection is available via _connect_ command. Manual connection can also be used for boards without XDS. Type _help connect_ or _help autoconnect_ for more info.

If the board is connected, the prompt will be green, otherwise, it will be red.

After connecting run plotter and prediction with following commands:

```bash
python mmwave-console.py
>> plot
>> predict
```

Use _Ctrl-C_ to stop this command.


### Collecting data

The console can be used for easy data collection. Use _log_ command to save gesture samples in .npz format in [mmwave/data/](./mmwave/data/) directory (or custom directory specified by `set_data_dir` command). If nothing is captured for more than a half a second, the command will automatically be stopped. _redraw_/_remove_ commands will redraw/remove the last captured sample.

```bash
python mmwave-console.py
>> listen
>> plot
>> set_data_dir /path/to/custom/data/dir
>> log up
>> log up
>> redraw up
>> remove up
>> log down
>> log ccw
```

### Training

```bash
python mmwave-console.py
>> set_data_dir /path/to/custom/data/dir
>> train
```

or

```bash
python mmwave_gesture/model.py
```

_Note: Default data dir is [mmwave_gesture/data](mmwave_gesture/data)._

### Selecting model
By default, conv2d model is used. Other models can be selected using _set_model_ option.
```bash
python mmwave-console.py
>> set_model conv1d
>> set_model lstm
>> set_model trans
```

### Help

Use help command to list all available commands and get documentation on them.

```bash
python mmwave-console.py
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
