"""Checkpoint loading and Teacher identity for training resumption."""

from pathlib import Path

import torch


def _load_checkpoint(path, kind, device):
    path = Path(path).absolute()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "schema_version", "stage", "model_kind", "model_state_dict",
        "config_snapshot", "data_config_snapshot", "run_metadata",
    }
    if kind == "student":
        required.update({"epoch", "best_epoch", "best_validation_pinball"})
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"{kind.capitalize()} checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["schema_version"] != 1:
        raise ValueError("unsupported AirTail checkpoint schema_version")
    if checkpoint["stage"] != "relation_kd_v2" or checkpoint["model_kind"] != kind:
        raise ValueError(f"checkpoint must contain an AirTail {kind}")
    return checkpoint, path


def load_teacher_checkpoint(path):
    return _load_checkpoint(path, "teacher", "cpu")


def load_student_checkpoint(path, model, model_config, data_config, device):
    checkpoint, path = _load_checkpoint(path, "student", device)
    if checkpoint["config_snapshot"] != model_config:
        raise ValueError("Student checkpoint model config does not match the current config")
    if checkpoint["data_config_snapshot"] != data_config:
        raise ValueError("Student checkpoint data config does not match the current config")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint, path


def validate_resume_checkpoint(path, stage, teacher_checkpoint, *, seed=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported AirTail resume checkpoint schema_version")
    if checkpoint.get("stage") != "relation_kd_v2" or checkpoint.get("model_kind") != stage:
        raise ValueError("resume checkpoint does not match the requested training stage")
    if seed is not None:
        saved_model_seed = checkpoint["config_snapshot"]["training"]["seed"]
        saved_loader_seed = checkpoint["data_config_snapshot"]["protocol"]["loader"]["seed"]
        if saved_model_seed != seed or saved_loader_seed != seed:
            raise ValueError("Saved training seeds differ from the requested seed. Start a new run.")
    if stage == "student":
        saved_teacher = checkpoint.get("teacher_state_dict")
        if saved_teacher is None:
            raise ValueError(
                "This Student training file does not contain its Teacher weights. "
                "Start a new Student run to create a resumable file."
            )
        current_teacher = teacher_checkpoint["model_state_dict"]
        if saved_teacher.keys() != current_teacher.keys() or any(
            saved_teacher[name].dtype != current_teacher[name].dtype
            or not torch.equal(saved_teacher[name], current_teacher[name])
            for name in saved_teacher
        ):
            raise ValueError("resume checkpoint used a different Teacher")
