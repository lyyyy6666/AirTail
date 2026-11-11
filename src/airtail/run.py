#!/usr/bin/env python
"""Command-line entry points for AirTail training, evaluation and prediction."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import yaml


from airtail.training.steps import (
    StudentTrainer,
    TeacherTrainer,
)
from airtail.training.fit import (
    fit_student,
    fit_teacher,
)
from airtail.evaluation.metrics import evaluate_qc_model
from airtail.data.bthsa_spatial import build_bthsa_spatial_bundle
from airtail.data.regional_spatial import build_regional_spatial_bundle
from airtail.data.factory import load_config
from airtail.training.reproducibility import set_reproducible_seed
from airtail.training.checkpoint import (
    load_teacher_checkpoint,
    load_student_checkpoint,
    validate_resume_checkpoint,
)
from airtail.models.pmg_rd.builder import (
    build_relation_loss,
    build_objective,
    build_airtail_student,
    build_airtail_teacher,
)


def _resolve_path(path):
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def _load_yaml(path):
    path = _resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("model config must contain a mapping")
    return value


def _teacher_relevant_config(config):
    relevant = copy.deepcopy(config)
    relation = relevant["relation_kd_v2"]
    relation.setdefault("teacher_relation_mode", "cross_evidence")
    for field in ("ablation_name", "distillation", "student_inputs", "student_relation_attention"):
        relation.pop(field, None)
    relevant.pop("stage", None)
    relevant.pop("head", None)
    relevant.get("taus", {}).pop("loss_weighting", None)
    return relevant


def _load_json(path):
    path = _resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = load_config(path)
    if not isinstance(value, dict):
        raise TypeError("data config must contain a mapping")
    return value


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {name}, but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _validate_runner_config(model_config):
    training = model_config["training"]
    if training["amp"] is not False:
        raise ValueError("AirTail runner requires training.amp=false")
    optimizer = model_config["optimizer"]
    if optimizer["name"] != "AdamW":
        raise ValueError("AirTail supports only the configured AdamW optimizer")
    batch_size = training["batch_size"]
    microbatch_size = training["microbatch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("training.batch_size must be a positive integer")
    if batch_size != 16:
        raise ValueError("AirTail requires training.batch_size=16")
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int):
        raise TypeError("training.microbatch_size must be a positive integer")
    if microbatch_size <= 0:
        raise ValueError("training.microbatch_size must be positive")
    if microbatch_size > batch_size:
        raise ValueError("training.microbatch_size must not exceed training.batch_size")
    if microbatch_size != 8:
        raise ValueError("AirTail requires training.microbatch_size=8")
    if "accumulation_steps" in training:
        raise ValueError("accumulation_steps is derived; configure only microbatch_size")


def _build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
    }
    for name in ("betas", "eps", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
        if name in optimizer_config:
            kwargs[name] = optimizer_config[name]
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _taus(model_config, device):
    values = torch.tensor(model_config["taus"]["fixed"], dtype=torch.float32, device=device)
    if values.ndim != 1 or values.numel() == 0 or not torch.all(values[1:] > values[:-1]):
        raise ValueError("taus.fixed must be a strictly increasing non-empty vector")
    return values


def _training_tau_weights(model_config, taus):
    mode = model_config["taus"].get("loss_weighting", "point_equal")
    if mode == "point_equal":
        return None
    if mode != "region_equal":
        raise ValueError("taus.loss_weighting must be point_equal or region_equal")
    tau_u = float(model_config["head"]["tau_u"])
    bulk = taus <= tau_u
    tail = taus > tau_u
    if not torch.any(bulk) or not torch.any(tail):
        raise ValueError("region_equal weighting requires bulk and tail quantiles")
    weights = torch.zeros_like(taus)
    weights[bulk] = 0.5 / int(bulk.sum().item())
    weights[tail] = 0.5 / int(tail.sum().item())
    return weights


def _validate_data_bundle(bundle, model_config, *, privileged=True):
    if len(bundle.station_ids) != int(model_config["num_nodes"]):
        raise ValueError("configured num_nodes does not match regional station order")
    if len(bundle.met_feature_names) != int(model_config["met_dim"]):
        raise ValueError("configured met_dim does not match regional history meteorology")
    expected_future = tuple(model_config["teacher"]["future_met_variables"])
    if tuple(bundle.future_met_feature_names) != (expected_future if privileged else ()):
        raise ValueError("future meteorology fields do not match the model config")
    if len(expected_future) != int(model_config["teacher"]["future_met_dim"]):
        raise ValueError("future_met_dim does not match configured variables")
    if {"PM2.5", "O3"} & set(expected_future):
        raise ValueError("future meteorology contains a pollutant variable")
    dataset = bundle.datasets["train"]
    window = model_config["horizon"]
    if dataset.history_steps != 24:
        raise ValueError("AirTail requires a 24-step history window")
    if int(window) != dataset.prediction_steps:
        raise ValueError("configured horizon does not match the data window")


def _build_spatial_bundle(data_config_path, **kwargs):
    config = _load_json(data_config_path)
    if config["dataset"] == "BTHSA":
        return build_bthsa_spatial_bundle(data_config_path, **kwargs)
    return build_regional_spatial_bundle(data_config_path, **kwargs)


def _evaluate_student_checkpoint(
    *, checkpoint_path, model_config, data_config, model, device, data_config_path,
    include_test=False,
    paper_metrics=False,
):
    """Evaluate one Student checkpoint on validation and optionally test."""
    checkpoint, checkpoint_path = load_student_checkpoint(
        checkpoint_path, model, model_config, data_config, device
    )
    included_splits = ("train", "validation")
    if include_test:
        included_splits += ("test",)
    eval_bundle = _build_spatial_bundle(
        data_config_path,
        check_files=True,
        loader_overrides={
            "batch_size": int(model_config["training"]["batch_size"]),
            "seed": int(model_config["training"]["seed"]),
        },
        included_splits=included_splits,
        future_met_variables=(),
    )
    _validate_data_bundle(eval_bundle, model_config, privileged=False)
    validation_metrics = evaluate_qc_model(
        model, eval_bundle.loaders["validation"], eval_bundle, device,
        model_config, "student", paper_metrics=paper_metrics
    )
    training_best = float(checkpoint["best_validation_pinball"])
    if abs(float(validation_metrics["standardized_pinball"]) - training_best) > 1e-5:
        raise RuntimeError(
            "best-checkpoint validation pinball is inconsistent with training: "
            f"training={training_best}, evaluator={validation_metrics['standardized_pinball']}"
        )
    artifact = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_pinball": training_best,
        "validation_metrics": validation_metrics,
    }
    if include_test:
        artifact["test_metrics"] = evaluate_qc_model(
            model, eval_bundle.loaders["test"], eval_bundle, device,
            model_config, "student", paper_metrics=paper_metrics
        )
    return artifact


def _prepare_paths(args, stage, teacher_checkpoint, seed):
    output_dir = _resolve_path(args.output_dir)
    if args.resume_checkpoint is None:
        if output_dir.exists():
            raise FileExistsError(f"fresh output directory already exists: {output_dir}")
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume requires an existing output directory: {output_dir}")
        validate_resume_checkpoint(args.resume_checkpoint, stage, teacher_checkpoint, seed=seed)
    return output_dir


def run(args):
    stage = args.stage
    if stage not in {"teacher", "student", "evaluate"}:
        raise ValueError("stage must be teacher, student, or evaluate")
    if stage == "evaluate":
        if getattr(args, "model_kind", "student") != "student":
            raise ValueError("AirTail evaluation currently supports only model-kind student")
        checkpoint_arg = getattr(args, "checkpoint", None)
        if checkpoint_arg is None:
            raise ValueError("evaluate stage requires --checkpoint")
        model_config_path = _resolve_path(args.model_config)
        data_config_path = _resolve_path(args.data_config)
        model_config = _load_yaml(model_config_path)
        data_config = _load_json(data_config_path)
        data_config["protocol"]["loader"]["seed"] = int(model_config["training"]["seed"])
        _validate_runner_config(model_config)
        device = resolve_device(args.device)
        model = build_airtail_student(model_config, data_config).to(device)
        artifact = _evaluate_student_checkpoint(
            checkpoint_path=checkpoint_arg,
            model_config=model_config,
            data_config=data_config,
            model=model,
            device=device,
            data_config_path=data_config_path,
            include_test=bool(getattr(args, "include_test", False)),
            paper_metrics=args.paper_metrics,
        )
        output_dir = _resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_dir / "best_relation_student_evaluation.json", artifact)
        result_path = output_dir / "relation_student_result.json"
        if result_path.is_file():
            # Make a previously completed training summary immediately useful
            # when evaluation is run as a standalone post-hoc stage.
            with result_path.open("r", encoding="utf-8") as handle:
                training_result = json.load(handle)
            training_result["best_validation_metrics"] = artifact["validation_metrics"]
            if "test_metrics" in artifact:
                training_result["test_metrics"] = artifact["test_metrics"]
            _atomic_write_json(result_path, training_result)
        return artifact
    if stage == "teacher" and args.teacher_checkpoint is not None:
        raise ValueError("Teacher stage does not accept --teacher-checkpoint")
    if stage == "student" and args.teacher_checkpoint is None:
        raise ValueError("Student stage requires --teacher-checkpoint")

    model_config_path = _resolve_path(args.model_config)
    data_config_path = _resolve_path(args.data_config)
    model_config = _load_yaml(model_config_path)
    data_config = _load_json(data_config_path)
    data_config["protocol"]["loader"]["seed"] = int(model_config["training"]["seed"])
    _validate_runner_config(model_config)
    device = resolve_device(args.device)
    teacher_checkpoint = None
    if stage == "student":
        teacher_checkpoint, teacher_checkpoint_path = load_teacher_checkpoint(args.teacher_checkpoint)
        if _teacher_relevant_config(teacher_checkpoint["config_snapshot"]) != _teacher_relevant_config(model_config):
            raise ValueError("Teacher model config does not match the current training config")
        if teacher_checkpoint["data_config_snapshot"] != data_config:
            raise ValueError("Teacher checkpoint data config snapshot does not match current data config")
    output_dir = _prepare_paths(args, stage, teacher_checkpoint, int(model_config["training"]["seed"]))

    set_reproducible_seed(int(model_config["training"]["seed"]))
    bundle = _build_spatial_bundle(
        data_config_path,
        check_files=True,
        loader_overrides={
            "batch_size": int(model_config["training"]["batch_size"]),
            "seed": int(model_config["training"]["seed"]),
        },
        included_splits=("train", "validation"),
        future_met_variables=tuple(model_config["teacher"]["future_met_variables"]),
    )
    _validate_data_bundle(bundle, model_config)
    coords = bundle.coords_km.to(device)
    neighbor_index = bundle.neighbor_index.to(device)
    taus = _taus(model_config, device)
    tau_weights = _training_tau_weights(model_config, taus)
    train_loader = bundle.loaders["train"]
    validation_loader = bundle.loaders["validation"]
    run_metadata = {
        "seed": int(model_config["training"]["seed"]),
        "device": str(device),
    }

    if stage == "student":
        run_metadata["teacher_checkpoint_path"] = str(teacher_checkpoint_path)

    if stage == "teacher":
        teacher = build_airtail_teacher(model_config, data_config).to(device)
        optimizer = _build_optimizer(teacher, model_config)
        trainer = TeacherTrainer(
            teacher,
            optimizer,
            coords=coords,
            neighbor_index=neighbor_index,
            taus=taus,
            grad_clip=float(model_config["training"]["grad_clip"]),
            microbatch_size=int(model_config["training"]["microbatch_size"]),
            loader_batch_size=int(model_config["training"]["batch_size"]),
        )
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)
        fit = fit_teacher(
            trainer, train_loader, validation_loader, output_dir,
            int(model_config["training"]["epochs"]),
            int(model_config["training"]["early_stopping_patience"]),
            copy.deepcopy(model_config), copy.deepcopy(data_config), run_metadata,
            resume_checkpoint=args.resume_checkpoint,
        )
        result_name = "relation_teacher_result.json"
    else:
        # Rebuild the Teacher from its saved configuration before loading weights.
        teacher_config = teacher_checkpoint["config_snapshot"]
        teacher_model = build_airtail_teacher(teacher_config, data_config).to(device)
        teacher_state_dict = teacher_checkpoint["model_state_dict"]
        teacher_model.load_state_dict(teacher_state_dict, strict=True)
        del teacher_state_dict
        del teacher_checkpoint
        student = build_airtail_student(model_config, data_config).to(device)
        optimizer = _build_optimizer(student, model_config)
        relation_loss = build_relation_loss(model_config)
        objective = build_objective(
            model_config, steps_per_epoch=len(train_loader)
        )
        trainer = StudentTrainer(
            student,
            teacher_model,
            optimizer,
            relation_loss,
            objective,
            coords=coords,
            neighbor_index=neighbor_index,
            taus=taus,
            tau_weights=tau_weights,
            grad_clip=float(model_config["training"]["grad_clip"]),
            microbatch_size=int(model_config["training"]["microbatch_size"]),
            loader_batch_size=int(model_config["training"]["batch_size"]),
        )
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)
        fit = fit_student(
            trainer, train_loader, validation_loader, output_dir,
            int(model_config["training"]["epochs"]),
            int(model_config["training"]["early_stopping_patience"]),
            copy.deepcopy(model_config), copy.deepcopy(data_config), run_metadata,
            resume_checkpoint=args.resume_checkpoint,
        )
        result_name = "relation_student_result.json"

    result = {
        "stage": stage,
        "model_kind": stage,
        "best_epoch": fit.best_epoch,
        "best_validation_pinball": fit.best_validation_pinball,
        "last_epoch": fit.last_epoch,
        "global_step": fit.global_step,
        "stopped_early": fit.stopped_early,
        "epochs_completed": fit.epochs_completed,
        "history": list(fit.history),
        "best_checkpoint_path": str(fit.best_checkpoint_path),
        "latest_checkpoint_path": str(fit.latest_checkpoint_path),
        "run_metadata": run_metadata,
    }
    if stage == "student":
        # Reload the selected checkpoint for standardized validation metrics.
        # Exploratory runs do not open the test split unless explicitly asked.
        if fit.best_checkpoint_path.is_file():
            evaluation_artifact = _evaluate_student_checkpoint(
                checkpoint_path=fit.best_checkpoint_path,
                model_config=model_config,
                data_config=data_config,
                model=student,
                device=device,
                data_config_path=data_config_path,
                include_test=bool(getattr(args, "include_test", False)),
            )
            result["best_validation_metrics"] = evaluation_artifact["validation_metrics"]
            if "test_metrics" in evaluation_artifact:
                result["test_metrics"] = evaluation_artifact["test_metrics"]
            _atomic_write_json(output_dir / "best_relation_student_evaluation.json", evaluation_artifact)
    _atomic_write_json(output_dir / result_name, result)
    return result


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _predict(args):
    device = resolve_device(args.device)
    # Training checkpoints contain optimizer and RNG state. Use trusted local files.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("schema_version"), checkpoint.get("stage"), checkpoint.get("model_kind")) != (1, "relation_kd_v2", "student"):
        raise ValueError("predict requires an AirTail Student training checkpoint")
    config = copy.deepcopy(checkpoint["config_snapshot"])
    if args.scan_backend:
        config["ssm"]["backend"] = args.scan_backend
    model = build_airtail_student(config, checkpoint["data_config_snapshot"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    fields = {
        "history_values", "history_mask", "history_met", "history_wind_uv",
        "history_time_features", "coords", "neighbor_index",
    }
    payload = torch.load(args.input, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError(f"predict input requires these tensor keys: {sorted(fields)}")
    if any(not isinstance(value, torch.Tensor) for value in payload.values()):
        raise TypeError("predict inputs must be tensors")
    expected = checkpoint["data_config_snapshot"]["protocol"]["window"]["history_steps"]
    if payload["history_values"].ndim != 4 or payload["history_values"].shape[1] != expected:
        raise ValueError(f"predict history_values must have shape [B,{expected},N,Fx]")
    if payload["history_mask"].dtype != torch.bool:
        raise ValueError("predict history_mask must have dtype bool")
    taus = _taus(config, device)
    with torch.inference_mode():
        prediction = model.deploy_forward(**payload, taus=taus)
    with open(args.output, "xb") as output:
        torch.save({"quantiles": prediction.quantiles.cpu(), "taus": taus.cpu()}, output)


def build_parser():
    parser = argparse.ArgumentParser(description="AirTail PM2.5 forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, stage in (("train-teacher", "teacher"), ("train-student", "student"), ("evaluate", "evaluate")):
        sub = commands.add_parser(command)
        sub.set_defaults(stage=stage, teacher_checkpoint=None, include_test=False)
        sub.add_argument("--model-config", "--config", required=True)
        sub.add_argument("--data-config", required=True)
        sub.add_argument("--output-dir", required=True)
        sub.add_argument("--device", default="cuda:0")
        if stage == "evaluate":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--paper-metrics", action="store_true")
            sub.add_argument("--split", choices=("validation", "test"), default="validation")
        else:
            sub.add_argument("--resume-checkpoint")
            if stage == "student":
                sub.add_argument("--teacher-checkpoint", required=True)
    sub = commands.add_parser("predict")
    sub.add_argument("--checkpoint", required=True)
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--device", default="cuda:0")
    sub.add_argument("--scan-backend", choices=("reference", "cuda"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "predict":
        _predict(args)
        return
    if args.command == "evaluate":
        args.include_test = args.split == "test"
    run(args)
