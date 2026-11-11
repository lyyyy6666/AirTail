"""YRD unit metadata is optional, while variables and time coordinates are required."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from airtail.data.regional_spatial import _load_panel


class YRDReadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "dataset_yrd.nc"
        self.values = np.arange(16, dtype=np.float32).reshape(8, 2)
        self.history_names = ("O3", "u100", "v100")
        with h5py.File(self.path, "w") as panel:
            panel["PM2.5"] = self.values
            for index, name in enumerate(self.history_names):
                panel[name] = self.values + index
            panel["station"] = np.array([b"1001A", b"1002A"])
            panel["time"] = np.arange(8, dtype=np.int64)
            panel["time"].attrs["units"] = "hours since 2016-01-01"
        stations = self.root / "stations_yrd.csv"
        stations.write_text("station_id,lon,lat\n1001A,120,31\n1002A,121,32\n")
        self.config = {
            "dataset": "YRD",
            "source": {"data_path": str(self.path), "stations_path": str(stations)},
            "selection": {"target_variable": "PM2.5"},
        }

    def test_missing_units_preserve_raw_values_and_station_order(self):
        target, met, timestamps, stations, _, _, units, _ = _load_panel(
            self.config, 6, self.history_names,
        )
        np.testing.assert_array_equal(target[..., 0], self.values[:6])
        np.testing.assert_array_equal(met[..., 0], self.values[:6])
        self.assertEqual(stations, ("1001A", "1002A"))
        self.assertEqual(units, (None, None, None))
        self.assertEqual(timestamps[0], np.datetime64("2016-01-01T00:00:00"))
        self.assertEqual(timestamps[-1], np.datetime64("2016-01-01T05:00:00"))

    def test_existing_units_are_retained_without_changing_data(self):
        before = _load_panel(self.config, 6, self.history_names)
        with h5py.File(self.path, "a") as panel:
            panel["O3"].attrs["units"] = "ug/m3"
            panel["u100"].attrs["units"] = "m/s"
            panel["v100"].attrs["units"] = "m/s"
        after = _load_panel(self.config, 6, self.history_names)
        for index in (0, 1, 2):
            np.testing.assert_array_equal(before[index], after[index])
        self.assertEqual(after[6], ("ug/m3", "m/s", "m/s"))

    def test_time_units_remain_required(self):
        with h5py.File(self.path, "a") as panel:
            del panel["time"].attrs["units"]
        with self.assertRaisesRegex(RuntimeError, "time.units"):
            _load_panel(self.config, 6, self.history_names)

    def test_missing_variable_remains_an_error(self):
        with h5py.File(self.path, "a") as panel:
            del panel["O3"]
        with self.assertRaises(KeyError):
            _load_panel(self.config, 6, self.history_names)
