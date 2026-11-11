"""Data files must exist and match the expected panel sizes."""

import tempfile
import unittest
from pathlib import Path

from airtail.data.factory import check_data_files


class DataFileTests(unittest.TestCase):
    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations"
            path.write_bytes(b"1234")
            manifest = {"files": [{"path": str(path), "bytes": 4}]}
            check_data_files(manifest)
            path.write_bytes(b"123")
            with self.assertRaisesRegex(ValueError, "file size"):
                check_data_files(manifest)

    def test_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations"
            with self.assertRaises(FileNotFoundError):
                check_data_files({"files": [{"path": str(path), "bytes": 4}]})
