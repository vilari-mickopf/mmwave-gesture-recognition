#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='mmwave',
      version='0.2',
      description='mmWave gesture recognition',
      url='http://github.com/f12markovic/mmwave-gesture-recognition',
      author='Filip Markovic',
      author_email='f12markovic@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['pyserial',
                        'PyQt5',
                        'pyqt5-tools',
                        'pyreadline',
                        'pynput',
                        'colorama',
                        'tqdm',
                        'pandas',
                        'numpy',
                        'multimethod',
                        'matplotlib',
                        'seaborn',
                        'scikit-learn',
                        'tensorflow'])
