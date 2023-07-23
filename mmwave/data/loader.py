#!/usr/bin/env python

from pathlib import Path

import numpy as np


class DataLoaderMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._loaders = {}

    def register(cls, ext):
        def inner(func):
            extensions = [ext] if not isinstance(ext, list) else ext
            cls._loaders.update({e: func for e in extensions})
            return func
        return inner


class DataLoader(metaclass=DataLoaderMeta):
    def __init__(self, path):
        self.path = path
        self.extension = Path(path).suffix.lstrip('.')

    def load(self):
        loader = self._loaders.get(self.extension)
        if loader is None:
            extensions = ', '.join(self._loaders.keys())
            raise ValueError(f'Unsupported file extension: "{self.extension}". '
                             f'Suported extensions: {extensions}.')
        return loader(self)


@DataLoader.register(['npy', 'npz'])
def _(self):
    return np.load(self.path, allow_pickle=True)['data']


# Add custom loaders here:
#  @DataLoader.register('csv')
#  def _(self):
#      return pd.read_csv(self.path)
