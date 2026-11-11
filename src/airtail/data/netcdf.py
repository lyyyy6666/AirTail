"""Raw NetCDF4/HDF5 access without automatic masking or scaling."""

import h5py
import numpy as np


class NetCDFReader:
    def __init__(self, path):
        self.file = h5py.File(path, "r")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def variable_shape(self, name):
        return self.file[name].shape

    def read_float(self, name, first_axis_stop=None):
        values = self.file[name]
        if first_axis_stop is None:
            return np.asarray(values[...], dtype=np.float32)
        if not 0 < first_axis_stop <= values.shape[0]:
            raise ValueError("first_axis_stop is outside the variable")
        return np.asarray(values[:first_axis_stop], dtype=np.float32)

    def read_double(self, name):
        return np.asarray(self.file[name][...], dtype=np.float64)

    def read_strings(self, name):
        return tuple(self.file[name].asstr()[:])

    def text_attribute(self, variable, attribute):
        try:
            value = self.file[variable].attrs[attribute]
        except KeyError as exc:
            raise RuntimeError(f"missing attribute {variable}.{attribute}") from exc
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
