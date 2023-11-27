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
    def load(self, path):
        extension = Path(path).suffix.lstrip('.')

        loader = self._loaders.get(extension)
        if loader is None:
            extensions = ', '.join(self._loaders.keys())
            raise ValueError(f'Unsupported file extension: "{extension}". '
                             f'Suported extensions: {extensions}.')
        return loader(path)

    @staticmethod
    def get_extensions():
        return list(DataLoader()._loaders.keys())


@DataLoader.register(['npy', 'npz'])
def _(path):
    return np.load(path, allow_pickle=True)['data']


# Add custom loaders here:
#  @DataLoader.register('csv')
#  def _(path):
#      return pd.read_csv(path)
