"""The published training settings use a single experiment seed."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from airtail import run
from airtail.training.checkpoint import validate_resume_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class SeedTests(unittest.TestCase):
    def test_paper_configurations_use_seed_2024(self):
        for stage in ("teacher", "student"):
            config = yaml.safe_load((ROOT / f"configs/{stage}.yaml").read_text())
            self.assertEqual(config["training"]["seed"], 2024)
        for path in (ROOT / "configs/datasets").glob("*.json"):
            config = json.loads(path.read_text())
            self.assertEqual(config["protocol"]["loader"]["seed"], 2024)

    def test_training_seed_controls_loader_even_with_a_different_data_default(self):
        config = json.loads((ROOT / "configs/datasets/bthsa.json").read_text())
        config["protocol"]["loader"]["seed"] = 17
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.json"
            data.write_text(json.dumps(config))
            with patch.object(run, "_build_spatial_bundle", side_effect=RuntimeError("capture loader")) as build:
                with self.assertRaisesRegex(RuntimeError, "capture loader"):
                    run.main([
                        "train-teacher", "--config", str(ROOT / "configs/teacher.yaml"),
                        "--data-config", str(data), "--output-dir", str(root / "teacher"),
                        "--device", "cpu",
                    ])
            self.assertEqual(build.call_args.kwargs["loader_overrides"]["seed"], 2024)

    def test_resume_rejects_a_saved_loader_seed_from_a_different_experiment(self):
        checkpoint = {
            "schema_version": 1,
            "stage": "relation_kd_v2",
            "model_kind": "teacher",
            "config_snapshot": {"training": {"seed": 2024}},
            "data_config_snapshot": {"protocol": {"loader": {"seed": 17}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Saved training seeds differ"):
                validate_resume_checkpoint(path, "teacher", None, seed=2024)
            checkpoint["data_config_snapshot"]["protocol"]["loader"]["seed"] = 2024
            torch.save(checkpoint, path)
            validate_resume_checkpoint(path, "teacher", None, seed=2024)
