import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


RELATION_KD_V2_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RelationKDV2FitResult:
    best_epoch: int
    best_validation_pinball: float
    last_epoch: int
    global_step: int
    stopped_early: bool
    epochs_completed: int
    history: tuple
    best_checkpoint_path: Path
    latest_checkpoint_path: Path


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _trainer_components(trainer, model_kind):
    if model_kind == "teacher":
        if hasattr(trainer, "student") or not hasattr(trainer, "teacher"):
            raise TypeError("Teacher fit requires a Teacher single-step Trainer")
        model = trainer.teacher
    else:
        if not hasattr(trainer, "student") or not hasattr(trainer, "teacher"):
            raise TypeError("Student fit requires a Student single-step Trainer")
        model = trainer.student
        _enforce_frozen_teacher(trainer)
    if not hasattr(trainer, "optimizer") or not hasattr(trainer, "global_step"):
        raise TypeError("single-step Trainer must expose optimizer and global_step")
    if not callable(getattr(trainer, "train_step", None)):
        raise TypeError("single-step Trainer must provide train_step")
    if not callable(getattr(trainer, "validation_step", None)):
        raise TypeError("single-step Trainer must provide validation_step")
    return model, trainer.optimizer


def _enforce_frozen_teacher(trainer):
    teacher = trainer.teacher
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    if teacher_ids & optimizer_ids:
        raise ValueError("Student optimizer must not contain Teacher parameters")


def _weighted_train_metrics(results, model_kind):
    denominator = sum(int(result.valid_count) for result in results)
    if denominator <= 0:
        raise ValueError("training pinball denominator must be positive")
    train_pinball = sum(
        float(result.pinball_numerator) for result in results
    ) / denominator
    if model_kind == "teacher":
        return {
            "train_pinball": train_pinball,
            "train_relation": 0.0,
            "train_total": train_pinball,
            "mean_relation_weight": 0.0,
        }
    return {
        "train_pinball": train_pinball,
        "train_relation": sum(
            float(result.relation_loss) * int(result.valid_count)
            for result in results
        )
        / denominator,
        "train_total": sum(
            float(result.total_loss) * int(result.valid_count)
            for result in results
        )
        / denominator,
        "mean_relation_weight": sum(
            float(result.relation_weight) * int(result.valid_count)
            for result in results
        )
        / denominator,
    }


def _run_train_epoch(trainer, train_loader, model_kind):
    results = [trainer.train_step(batch) for batch in train_loader]
    if not results:
        raise ValueError("training loader is empty")
    return _weighted_train_metrics(results, model_kind)


def _run_validation_epoch(trainer, validation_loader):
    numerator = 0.0
    denominator = 0
    batches = 0
    for batch in validation_loader:
        result = trainer.validation_step(batch)
        numerator += float(result.pinball_numerator)
        denominator += int(result.pinball_denominator)
        batches += 1
    if batches == 0:
        raise ValueError("validation loader is empty")
    if denominator <= 0:
        raise ValueError("validation pinball denominator must be positive")
    return {
        "standardized_pinball": numerator / denominator,
        "pinball_numerator": numerator,
        "pinball_denominator": denominator,
        "batches": batches,
    }


def _capture_rng_state(train_loader):
    generator = getattr(train_loader, "generator", None)
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "loader_generator_state": (
            generator.get_state() if generator is not None else None
        ),
    }


def _restore_rng_state(state, train_loader):
    if state is None:
        return
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    saved_generator_state = state["loader_generator_state"]
    if saved_generator_state is not None:
        generator = getattr(train_loader, "generator", None)
        if generator is None:
            raise ValueError("resume checkpoint expects a train loader generator")
        generator.set_state(saved_generator_state.cpu())


def _atomic_save(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint(
    *,
    stage,
    model_kind,
    model,
    optimizer,
    epoch,
    global_step,
    best_epoch,
    best_validation_pinball,
    epochs_without_improvement,
    validation_metrics,
    history,
    config_snapshot,
    data_config_snapshot,
    run_metadata,
    train_loader,
):
    return {
        "schema_version": RELATION_KD_V2_CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "model_kind": model_kind,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_epoch": int(best_epoch),
        "best_validation_pinball": float(best_validation_pinball),
        "epochs_without_improvement": int(epochs_without_improvement),
        "validation_metrics": deepcopy(validation_metrics),
        "history": deepcopy(history),
        "config_snapshot": deepcopy(config_snapshot),
        "data_config_snapshot": deepcopy(data_config_snapshot),
        "run_metadata": deepcopy(run_metadata),
        "rng_state": _capture_rng_state(train_loader),
    }


def _load_resume(path, model, optimizer, trainer, train_loader, stage, model_kind):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "schema_version",
        "stage",
        "model_kind",
        "epoch",
        "next_epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "best_epoch",
        "best_validation_pinball",
        "epochs_without_improvement",
        "validation_metrics",
        "history",
        "config_snapshot",
        "data_config_snapshot",
        "run_metadata",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Relation-KD V2 checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["schema_version"] != RELATION_KD_V2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported Relation-KD V2 checkpoint schema_version: "
            f"{checkpoint['schema_version']}"
        )
    if checkpoint["stage"] != stage:
        raise ValueError(
            f"checkpoint stage {checkpoint['stage']!r} does not match {stage!r}"
        )
    if checkpoint["model_kind"] != model_kind:
        raise ValueError(
            "checkpoint model_kind "
            f"{checkpoint['model_kind']!r} does not match {model_kind!r}"
        )
    if int(checkpoint["next_epoch"]) != int(checkpoint["epoch"]) + 1:
        raise ValueError("checkpoint next_epoch must follow its completed epoch")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.global_step = int(checkpoint["global_step"])
    _restore_rng_state(checkpoint.get("rng_state"), train_loader)
    return checkpoint


def _fit(
    *,
    trainer,
    train_loader,
    validation_loader,
    output_directory,
    epochs,
    early_stopping_patience,
    config_snapshot,
    data_config_snapshot,
    run_metadata,
    resume_checkpoint,
    stage,
    model_kind,
    best_name,
    latest_name,
):
    epochs = _positive_integer(epochs, "epochs")
    early_stopping_patience = _positive_integer(
        early_stopping_patience, "early_stopping_patience"
    )
    model, optimizer = _trainer_components(trainer, model_kind)
    output_directory = Path(output_directory)
    best_path = output_directory / best_name
    latest_path = output_directory / latest_name

    best_epoch = None
    best_validation_pinball = float("inf")
    epochs_without_improvement = 0
    history = []
    start_epoch = 1

    if resume_checkpoint is not None:
        checkpoint = _load_resume(
            resume_checkpoint,
            model,
            optimizer,
            trainer,
            train_loader,
            stage,
            model_kind,
        )
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_pinball = float(checkpoint["best_validation_pinball"])
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )
        history = deepcopy(checkpoint["history"])
        start_epoch = int(checkpoint["next_epoch"])
        if model_kind == "student":
            _enforce_frozen_teacher(trainer)

    stopped_early = False
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_train_epoch(trainer, train_loader, model_kind)
        validation_metrics = _run_validation_epoch(trainer, validation_loader)
        validation_pinball = validation_metrics["standardized_pinball"]
        improved = validation_pinball < best_validation_pinball
        if improved:
            best_epoch = epoch
            best_validation_pinball = validation_pinball
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "global_step": int(trainer.global_step),
            **train_metrics,
            "validation_standardized_pinball": float(validation_pinball),
            "improved": bool(improved),
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(record)

        checkpoint = _checkpoint(
            stage=stage,
            model_kind=model_kind,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=trainer.global_step,
            best_epoch=best_epoch,
            best_validation_pinball=best_validation_pinball,
            epochs_without_improvement=epochs_without_improvement,
            validation_metrics=validation_metrics,
            history=history,
            config_snapshot=config_snapshot,
            data_config_snapshot=data_config_snapshot,
            run_metadata=run_metadata,
            train_loader=train_loader,
        )
        if model_kind == "student":
            checkpoint["teacher_state_dict"] = {
                name: value.detach().cpu()
                for name, value in trainer.teacher.state_dict().items()
            }
        if improved:
            _atomic_save(checkpoint, best_path)
        _atomic_save(checkpoint, latest_path)

        if epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    if best_epoch is None:
        raise RuntimeError("training produced no valid validation checkpoint")
    last_epoch = history[-1]["epoch"]
    return RelationKDV2FitResult(
        best_epoch=best_epoch,
        best_validation_pinball=best_validation_pinball,
        last_epoch=last_epoch,
        global_step=int(trainer.global_step),
        stopped_early=stopped_early,
        epochs_completed=len(history),
        history=tuple(deepcopy(history)),
        best_checkpoint_path=best_path,
        latest_checkpoint_path=latest_path,
    )


def fit_teacher(
    trainer,
    train_loader,
    validation_loader,
    output_directory,
    epochs,
    early_stopping_patience,
    config_snapshot,
    data_config_snapshot,
    run_metadata,
    resume_checkpoint=None,
):
    return _fit(
        trainer=trainer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_directory=output_directory,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        config_snapshot=config_snapshot,
        data_config_snapshot=data_config_snapshot,
        run_metadata=run_metadata,
        resume_checkpoint=resume_checkpoint,
        stage="relation_kd_v2",
        model_kind="teacher",
        best_name="best_relation_teacher.pt",
        latest_name="latest_relation_teacher.pt",
    )


def fit_student(
    trainer,
    train_loader,
    validation_loader,
    output_directory,
    epochs,
    early_stopping_patience,
    config_snapshot,
    data_config_snapshot,
    run_metadata,
    resume_checkpoint=None,
):
    return _fit(
        trainer=trainer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_directory=output_directory,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        config_snapshot=config_snapshot,
        data_config_snapshot=data_config_snapshot,
        run_metadata=run_metadata,
        resume_checkpoint=resume_checkpoint,
        stage="relation_kd_v2",
        model_kind="student",
        best_name="best_relation_student.pt",
        latest_name="latest_relation_student.pt",
    )
