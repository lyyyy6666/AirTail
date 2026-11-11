from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from airtail.models.horizon_decoder import FixedCapacityHorizonDecoder
from airtail.models.backbone import QuantileBackbone

from .discovery import TemporalEvolutionDiscovery, VariableImpactDiscovery
from .losses import (
    TailEffectiveRelationDistillationLoss,
    compose_relation_kd_objective,
    linear_relation_distillation_weight,
)
from .relation_attention import RegionScaleRelationAttention
from .student import AirTailStudent, StudentRelationPath, StudentRelationPredictor
from .teacher import (
    AirTailTeacher,
    TeacherRelationGenerator,
    TeacherRelationPath,
)


def _future_config(model):
    if "future_conditioning" in model:
        return model["future_conditioning"]
    return model["teacher"]


def _load_config(config, name):
    if isinstance(config, dict):
        return deepcopy(config)
    path = Path(config)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
        else:
            value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a mapping")
    return value


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _relation_encoder_config(model):
    config = model["relation_kd_v2"].get("relation_encoder")
    if config is None:
        # The frozen Relation-KD V2 config predates this experimental switch.
        return "ktd", "state"
    if not isinstance(config, dict):
        raise TypeError("relation_kd_v2.relation_encoder must be a mapping")
    encoder_type = config["type"]
    if encoder_type not in {"ktd", "scpee"}:
        raise ValueError("relation_kd_v2.relation_encoder.type must be ktd or scpee")
    query_source = config.get("query_source", "state")
    if query_source not in {"state", "learned"}:
        raise ValueError(
            "relation_kd_v2.relation_encoder.query_source must be state or learned"
        )
    if encoder_type == "ktd" and query_source != "state":
        raise ValueError("relation_encoder.query_source applies only to scpee")
    return encoder_type, query_source


def _validate_training_protocol(model_config):
    training = model_config["training"]
    _non_negative_integer(training["seed"], "training.seed")
    _positive_integer(training["batch_size"], "training.batch_size")
    _positive_integer(training["epochs"], "training.epochs")
    _positive_integer(
        training["early_stopping_patience"],
        "training.early_stopping_patience",
    )
    if isinstance(training["amp"], bool) is False:
        raise TypeError("training.amp must be a bool")
    learning_rate = model_config["optimizer"]["learning_rate"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("optimizer.learning_rate must be a positive number")
    if not 0.0 < float(learning_rate) < float("inf"):
        raise ValueError("optimizer.learning_rate must be a positive finite number")


def _validated_inputs(model_config, data_config=None):
    model = _load_config(model_config, "model_config")
    data = None if data_config is None else _load_config(data_config, "data_config")
    _validate_training_protocol(model)

    d_model = _positive_integer(model["hidden_dim"], "hidden_dim")
    horizon = _positive_integer(model["horizon"], "horizon")
    kernels = model["temporal"]["kernels"]
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("temporal.kernels must be a non-empty list")
    num_scales = len(kernels)

    decoder = FixedCapacityHorizonDecoder(
        d_model=d_model,
        num_scales=num_scales,
        horizon=horizon,
        scale_fusion=model["decoder"]["scale_fusion"],
    )
    num_regions = decoder.num_regions
    num_tokens = num_regions * num_scales
    if num_tokens != 12:
        raise ValueError(
            "Relation-KD V2 requires num_regions * num_scales == 12"
        )

    architecture = model["relation_kd_v2"]["architecture"]
    _relation_encoder_config(model)
    relation_mode = model["relation_kd_v2"].get(
        "conditioned_relation_mode",
        model["relation_kd_v2"].get("teacher_relation_mode", "cross_evidence"),
    )
    if relation_mode not in {"cross_evidence", "temporal_only", "variable_only", "additive_fusion"}:
        raise ValueError("relation_kd_v2.teacher_relation_mode is invalid")
    num_heads = _positive_integer(
        architecture["num_heads"],
        "relation_kd_v2.architecture.num_heads",
    )
    if d_model % num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")

    legacy = model["distillation"]
    if legacy["curve_kd"] is not False:
        raise ValueError("distillation.curve_kd must be false")
    if legacy["reliability_weighting"] is not False:
        raise ValueError("distillation.reliability_weighting must be false")
    if legacy["lambda_kd"] != 0.0 or legacy["lambda_slope"] != 0.0:
        raise ValueError("legacy distillation weights must be zero")
    future = _future_config(model)
    if future["mode"] != "relation_kd_v2":
        raise ValueError("future conditioning mode must be relation_kd_v2")
    if future["future_spatial"] is not False:
        raise ValueError("future conditioning future_spatial must be false")
    distillation = model["relation_kd_v2"]["distillation"]
    if not isinstance(distillation.get("row_centering", True), bool):
        raise TypeError("relation_kd_v2.distillation.row_centering must be a bool")
    if not isinstance(future.get("use_future_meteorology", True), bool):
        raise TypeError("future conditioning use_future_meteorology must be a bool")
    if not isinstance(
        model["relation_kd_v2"].get("student_relation_attention", True),
        bool,
    ):
        raise TypeError(
            "relation_kd_v2.student_relation_attention must be a bool"
        )

    return model, data, d_model, horizon, num_regions, num_scales, num_tokens


def _history_weather_contract(model, data):
    if data is None:
        raise TypeError("data_config is required to build the teacher")
    history_variables = data["qc_mssm_spatial"]["history_met_variables"]
    future = _future_config(model)
    future_variables = future["future_met_variables"]
    if not isinstance(history_variables, list) or not isinstance(future_variables, list):
        raise TypeError("weather variable definitions must be lists")
    if len(history_variables) != model["met_dim"]:
        raise ValueError("history_met_variables must match met_dim")
    if len(future_variables) != future["future_met_dim"]:
        raise ValueError("future_met_variables must match future_met_dim")

    indices = []
    for name in future_variables:
        if history_variables.count(name) != 1:
            raise ValueError(
                f"future weather variable {name!r} must occur exactly once in "
                "history_met_variables"
            )
        indices.append(history_variables.index(name))
    if len(set(future_variables)) != len(future_variables):
        raise ValueError("future_met_variables must be unique")
    return tuple(indices), len(history_variables), len(future_variables)


def _build_attention(architecture, d_model, num_heads, num_tokens):
    return RegionScaleRelationAttention(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=architecture["dim_feedforward"],
        dropout=architecture["dropout"],
        activation=architecture["activation"],
        num_tokens=num_tokens,
        group_chunk_size=architecture["group_chunk_size"],
        relation_gain_init=architecture["relation_gain_init"],
        block_gain_init=architecture["block_gain_init"],
    )


def _student_backbone_contract(model, data):
    if data is None:
        raise TypeError("data_config is required to build the student")
    available = data["qc_mssm_spatial"]["history_met_variables"]
    inputs = model["relation_kd_v2"].get("student_inputs", {})
    if not isinstance(inputs, dict):
        raise TypeError("relation_kd_v2.student_inputs must be a mapping")
    selected = inputs.get("history_met_variables", available)
    if not isinstance(selected, list):
        raise TypeError("student_inputs.history_met_variables must be a list")
    if len(set(selected)) != len(selected):
        raise ValueError("student history variables must be unique")
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"unknown Student history variables: {unknown}")
    use_time = inputs.get("use_time_features", True)
    use_spatial = inputs.get("use_spatial", True)
    if not isinstance(use_time, bool) or not isinstance(use_spatial, bool):
        raise TypeError("student input switches must be bool values")

    student_model = deepcopy(model)
    indices = tuple(available.index(name) for name in selected)
    student_model["met_dim"] = len(indices)
    if not use_time:
        student_model["time_dim"] = 0
    if not use_spatial:
        student_model["spatial"]["mode"] = "none"
    return student_model, indices


def _build_conditioned_model(model_config, data_config, model_class):
    model, data, d_model, horizon, _, _, num_tokens = _validated_inputs(
        model_config,
        data_config,
    )
    architecture = model["relation_kd_v2"]["architecture"]
    num_heads = architecture["num_heads"]
    history_length = data["protocol"]["window"]["history_steps"]
    indices, num_covariates, num_weather_variables = _history_weather_contract(
        model,
        data,
    )

    backbone = QuantileBackbone(deepcopy(model))
    temporal_discovery = TemporalEvolutionDiscovery(
        d_model=d_model,
        num_weather_variables=num_weather_variables,
        history_length=history_length,
        prediction_length=horizon,
        dropout=architecture["dropout"],
        group_chunk_size=architecture["group_chunk_size"],
        use_future_meteorology=_future_config(model).get(
            "use_future_meteorology", True
        ),
    )
    variable_discovery = VariableImpactDiscovery(
        d_model=d_model,
        num_covariates=num_covariates,
        history_length=history_length,
        prediction_length=horizon,
        dropout=architecture["dropout"],
        group_chunk_size=architecture["group_chunk_size"],
    )
    encoder_type, query_source = _relation_encoder_config(model)
    relation_generator = TeacherRelationGenerator(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=architecture["token_distiller_num_layers"],
        dim_feedforward=architecture["dim_feedforward"],
        dropout=architecture["dropout"],
        activation=architecture["activation"],
        num_tokens=num_tokens,
        group_chunk_size=architecture["group_chunk_size"],
        relation_encoder_type=encoder_type,
        relation_query_source=query_source,
    )
    relation_generator.correlation.mode = model["relation_kd_v2"].get(
        "conditioned_relation_mode",
        model["relation_kd_v2"].get("teacher_relation_mode", "cross_evidence"),
    )
    relation_attention = _build_attention(
        architecture,
        d_model,
        num_heads,
        num_tokens,
    )
    relation_path = TeacherRelationPath(
        temporal_discovery=temporal_discovery,
        variable_discovery=variable_discovery,
        relation_generator=relation_generator,
        relation_attention=relation_attention,
    )
    return model_class(
        backbone=backbone,
        relation_path=relation_path,
        history_weather_indices=indices,
    )


def build_airtail_teacher(model_config, data_config):
    """Build the privileged AirTail model."""
    return _build_conditioned_model(model_config, data_config, AirTailTeacher)




def build_airtail_student(model_config, data_config=None):
    """Build an independent history-only Relation-KD V2 student."""
    model, data, d_model, _, _, _, num_tokens = _validated_inputs(
        model_config,
        data_config,
    )
    architecture = model["relation_kd_v2"]["architecture"]
    predictor_config = model["relation_kd_v2"]["student_predictor"]
    num_heads = architecture["num_heads"]

    student_model, history_met_indices = _student_backbone_contract(model, data)
    backbone = QuantileBackbone(student_model)
    relation_predictor = StudentRelationPredictor(
        d_model=d_model,
        num_heads=num_heads,
        relation_rank=predictor_config["relation_rank"],
        mlp_hidden_dim=predictor_config["mlp_hidden_dim"],
        dropout=architecture["dropout"],
        activation=architecture["activation"],
        num_tokens=num_tokens,
        group_chunk_size=architecture["group_chunk_size"],
    )
    relation_attention = _build_attention(
        architecture,
        d_model,
        num_heads,
        num_tokens,
    )
    relation_path = StudentRelationPath(
        relation_predictor=relation_predictor,
        relation_attention=relation_attention,
    )
    return AirTailStudent(
        backbone=backbone,
        relation_path=relation_path,
        history_met_indices=history_met_indices,
        relation_attention_enabled=model["relation_kd_v2"].get(
            "student_relation_attention", True
        ),
    )


def build_relation_loss(model_config):
    """Build the configured tail-effective relation distillation loss."""
    model, _, _, _, num_regions, num_scales, _ = _validated_inputs(model_config)
    distillation = model["relation_kd_v2"]["distillation"]
    loss = TailEffectiveRelationDistillationLoss(
        beta=distillation["smooth_l1_beta"],
        query_region_indices=distillation["query_region_indices"],
        num_regions=num_regions,
        num_scales=num_scales,
    )
    loss.row_centering = distillation.get("row_centering", True)
    return loss


@dataclass(frozen=True)
class RelationKDV2Objective:
    lambda_rel: float
    warmup_epochs: int
    steps_per_epoch: int | None

    @property
    def warmup_steps(self):
        if self.steps_per_epoch is None:
            return None
        return self.warmup_epochs * self.steps_per_epoch

    def relation_weight(self, *, global_step, warmup_steps=None):
        if warmup_steps is None:
            warmup_steps = self.warmup_steps
        if warmup_steps is None:
            raise ValueError(
                "warmup_steps is unknown; pass steps_per_epoch to the builder "
                "or warmup_steps to relation_weight"
            )
        return linear_relation_distillation_weight(
            lambda_rel=self.lambda_rel,
            global_step=global_step,
            warmup_steps=warmup_steps,
        )

    def __call__(
        self,
        pinball_loss,
        relation_loss,
        *,
        global_step,
        warmup_steps=None,
    ):
        relation_weight = self.relation_weight(
            global_step=global_step,
            warmup_steps=warmup_steps,
        )
        return compose_relation_kd_objective(
            pinball_loss,
            relation_loss,
            relation_weight=relation_weight,
        )


def build_objective(model_config, *, steps_per_epoch=None):
    """Preserve epoch warmup unless the caller supplies optimizer-step cadence."""
    model, _, _, _, _, _, _ = _validated_inputs(model_config)
    distillation = model["relation_kd_v2"]["distillation"]
    warmup_epochs = _non_negative_integer(
        distillation["relation_warmup_epochs"],
        "relation_kd_v2.distillation.relation_warmup_epochs",
    )
    if steps_per_epoch is not None:
        steps_per_epoch = _positive_integer(steps_per_epoch, "steps_per_epoch")
    return RelationKDV2Objective(
        lambda_rel=distillation["lambda_rel"],
        warmup_epochs=warmup_epochs,
        steps_per_epoch=steps_per_epoch,
    )
