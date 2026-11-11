"""Check model loading and direct Teacher parameter comparison."""

import tempfile
import unittest
from pathlib import Path

import torch

from airtail.training.checkpoint import (
    load_student_checkpoint,
    load_teacher_checkpoint,
    validate_resume_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def _checkpoint(self, kind, model):
        return {
            "schema_version": 1,
            "stage": "relation_kd_v2",
            "model_kind": kind,
            "model_state_dict": model.state_dict(),
            "config_snapshot": {"hidden_dim": 2},
            "data_config_snapshot": {"dataset": "BTHSA"},
            "run_metadata": {},
            "epoch": 1,
            "best_epoch": 1,
            "best_validation_pinball": 0.1,
        }

    def test_student_load_preserves_weights_and_rejects_missing_parameters(self):
        model = torch.nn.Linear(2, 1)
        checkpoint = self._checkpoint("student", model)
        path = self.root / "student.pt"
        torch.save(checkpoint, path)
        restored = torch.nn.Linear(2, 1)
        load_student_checkpoint(
            path, restored, checkpoint["config_snapshot"],
            checkpoint["data_config_snapshot"], "cpu",
        )
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, restored.state_dict()[name], rtol=0, atol=0)
        checkpoint["model_state_dict"].pop("bias")
        torch.save(checkpoint, path)
        with self.assertRaises(RuntimeError):
            load_student_checkpoint(
                path, restored, checkpoint["config_snapshot"],
                checkpoint["data_config_snapshot"], "cpu",
            )

    def test_resume_compares_teacher_parameters(self):
        teacher = self._checkpoint("teacher", torch.nn.Linear(2, 1))
        resume = self.root / "student.pt"
        torch.save({
            "schema_version": 1, "stage": "relation_kd_v2", "model_kind": "student",
            "teacher_state_dict": {
                name: value.clone() for name, value in teacher["model_state_dict"].items()
            },
        }, resume)
        validate_resume_checkpoint(resume, "student", teacher)
        teacher["model_state_dict"]["weight"] += 1
        with self.assertRaisesRegex(ValueError, "different Teacher"):
            validate_resume_checkpoint(resume, "student", teacher)

    def test_earlier_student_file_loads_for_evaluation_but_requires_teacher_for_resume(self):
        model = torch.nn.Linear(2, 1)
        checkpoint = self._checkpoint("student", model)
        path = self.root / "student.pt"
        torch.save(checkpoint, path)
        load_student_checkpoint(
            path, model, checkpoint["config_snapshot"], checkpoint["data_config_snapshot"], "cpu",
        )
        with self.assertRaisesRegex(ValueError, "does not contain its Teacher weights"):
            validate_resume_checkpoint(path, "student", self._checkpoint("teacher", model))

    def test_load_checks_role_and_data_config(self):
        model = torch.nn.Linear(2, 1)
        path = self.root / "model.pt"
        checkpoint = self._checkpoint("student", model)
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "AirTail teacher"):
            load_teacher_checkpoint(path)
        with self.assertRaisesRegex(ValueError, "data config"):
            load_student_checkpoint(
                path, model, checkpoint["config_snapshot"], {"dataset": "YRD"}, "cpu",
            )
