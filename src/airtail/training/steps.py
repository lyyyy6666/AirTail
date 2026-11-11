from dataclasses import dataclass

import torch

from airtail.training.pinball import weighted_integrated_pinball_parts


@dataclass(frozen=True)
class RelationKDV2StepResult:
    total_loss: float
    pinball_loss: float
    relation_loss: float
    relation_weight: float
    valid_count: int
    global_step_before: int
    global_step_after: int
    pinball_numerator: float
    pinball_denominator: int
    relation_numerator: float = 0.0
    relation_denominator: int = 0


def _validate_global_step(global_step):
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("global_step must be a non-negative integer")
    if global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
    return global_step


def _validate_static_inputs(coords, neighbor_index, taus, grad_clip):
    for name, value in (
        ("coords", coords),
        ("neighbor_index", neighbor_index),
        ("taus", taus),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
    if taus.ndim != 1 or taus.numel() == 0:
        raise ValueError("taus must be a non-empty vector")
    if not taus.is_floating_point():
        raise TypeError("taus must use a floating point dtype")
    if not torch.all(taus[1:] > taus[:-1]):
        raise ValueError("taus must be strictly increasing")
    if grad_clip is not None:
        if isinstance(grad_clip, bool) or not isinstance(grad_clip, (int, float)):
            raise TypeError("grad_clip must be None or a positive number")
        if not 0.0 < float(grad_clip) < float("inf"):
            raise ValueError("grad_clip must be None or a positive finite number")


def _validate_microbatch_size(microbatch_size, loader_batch_size=None):
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int):
        raise TypeError("microbatch_size must be a positive integer")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    if loader_batch_size is not None:
        if isinstance(loader_batch_size, bool) or not isinstance(loader_batch_size, int):
            raise TypeError("loader_batch_size must be a positive integer")
        if loader_batch_size <= 0:
            raise ValueError("loader_batch_size must be positive")
        if microbatch_size > loader_batch_size:
            raise ValueError("microbatch_size must not exceed loader_batch_size")
    return microbatch_size


def _batch_size(batch):
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("batch must contain a batched Tensor")


def _slice_batch(batch, start, end, batch_size):
    sliced = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def iter_microbatches(batch, microbatch_size):
    """Yield CPU-preserving slices of one loader batch."""
    batch_size = _batch_size(batch)
    _validate_microbatch_size(microbatch_size)
    for start in range(0, batch_size, microbatch_size):
        yield _slice_batch(batch, start, min(start + microbatch_size, batch_size), batch_size)


def derive_accumulation_steps(actual_batch_size, microbatch_size):
    if isinstance(actual_batch_size, bool) or not isinstance(actual_batch_size, int):
        raise TypeError("actual_batch_size must be a positive integer")
    if actual_batch_size <= 0:
        raise ValueError("actual_batch_size must be positive")
    _validate_microbatch_size(microbatch_size)
    return (actual_batch_size + microbatch_size - 1) // microbatch_size


def _move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return getattr(torch.Tensor, "to")(value, device)
    if isinstance(value, dict):
        return {key: _move_batch(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_batch(item, device) for item in value)
    if isinstance(value, list):
        return [_move_batch(item, device) for item in value]
    return value


def _history_inputs(batch, coords, neighbor_index, taus):
    required = {
        "history",
        "history_mask",
        "history_met",
        "history_wind_uv",
        "history_time_features",
        "target",
        "target_mask",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"BTHSA QC batch is missing keys: {missing}")
    time_features = batch["history_time_features"]
    if time_features.ndim != 4 or time_features.shape[-1] != 2:
        raise ValueError("history_time_features must have shape [B,L,N,2]")
    shared_time = time_features[:, :, :1].expand_as(time_features)
    if not torch.equal(time_features, shared_time):
        raise ValueError("calendar features must be identical across stations")
    return {
        "history_values": batch["history"],
        "history_mask": batch["history_mask"],
        "history_met": batch["history_met"],
        "history_wind_uv": batch["history_wind_uv"],
        "history_time_features": time_features[:, :, 0],
        "coords": coords,
        "neighbor_index": neighbor_index,
        "taus": taus,
    }


def _teacher_inputs(
    batch, coords, neighbor_index, taus, use_future_meteorology=True
):
    inputs = _history_inputs(batch, coords, neighbor_index, taus)
    if not use_future_meteorology:
        return {**inputs, "future_met": None, "future_met_mask": None}
    if "future_met" not in batch or "future_met_mask" not in batch:
        raise ValueError("AirTailTeacher requires future_met and future_met_mask")
    return {
        **inputs,
        "future_met": batch["future_met"],
        "future_met_mask": batch["future_met_mask"],
    }


def _target_valid_mask(batch):
    target = batch["target"]
    target_mask = batch["target_mask"]
    if target.ndim != 4 or target.shape[-1] != 1:
        raise ValueError("target must have shape [B,H,N,1]")
    if target_mask.shape != target.shape or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean and match target")
    return target_mask.any(dim=-1)


def _relation_group_mask(batch, use_future_meteorology=True):
    target_valid = _target_valid_mask(batch)
    if not use_future_meteorology:
        return target_valid
    future_valid = batch["future_met_mask"]
    if future_valid.shape != target_valid.shape + (1,):
        raise ValueError("future_met_mask must have shape [B,H,N,1]")
    if future_valid.dtype != torch.bool:
        raise TypeError("future_met_mask must have bool dtype")
    return target_valid & future_valid[..., 0]


def _relation_denominator(valid_mask, num_heads, relation_loss):
    if not hasattr(relation_loss, "query_rows"):
        relation_loss = relation_loss.inner
    groups = int(valid_mask.sum().item())
    query_count = len(relation_loss.query_rows)
    num_tokens = relation_loss.num_tokens
    return groups * int(num_heads) * query_count * num_tokens


def _pinball_parts(prediction, batch, taus, tau_weights=None):
    numerator_tensor, denominator_tensor, pinball = weighted_integrated_pinball_parts(
        prediction.quantiles,
        batch["target"],
        batch["target_mask"],
        taus,
        tau_weights,
    )
    valid_count = int(denominator_tensor.item())
    numerator = float(numerator_tensor.detach().item())
    return pinball, valid_count, numerator


def _check_finite(loss, name):
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"{name} is non-finite")


def _step_result(
    total_loss,
    pinball_loss,
    relation_loss,
    relation_weight,
    valid_count,
    global_step_before,
    global_step_after,
    pinball_numerator,
    relation_numerator=0.0,
    relation_denominator=0,
):
    return RelationKDV2StepResult(
        total_loss=float(total_loss.detach().item()),
        pinball_loss=float(pinball_loss.detach().item()),
        relation_loss=float(relation_loss.detach().item()),
        relation_weight=float(relation_weight),
        valid_count=valid_count,
        global_step_before=global_step_before,
        global_step_after=global_step_after,
        pinball_numerator=pinball_numerator,
        pinball_denominator=valid_count,
        relation_numerator=relation_numerator,
        relation_denominator=relation_denominator,
    )


class TeacherTrainer:
    def __init__(
        self,
        teacher,
        optimizer,
        *,
        coords,
        neighbor_index,
        taus,
        grad_clip=1.0,
        global_step=0,
        microbatch_size=8,
        loader_batch_size=None,
        scheduler=None,
    ):
        _validate_static_inputs(coords, neighbor_index, taus, grad_clip)
        parameters = tuple(teacher.parameters())
        if not parameters or not all(parameter.requires_grad for parameter in parameters):
            raise ValueError("AirTailTeacher must be fully trainable")
        self.teacher = teacher
        self.optimizer = optimizer
        self.coords = coords
        self.neighbor_index = neighbor_index
        self.taus = taus
        self.grad_clip = grad_clip
        self.global_step = _validate_global_step(global_step)
        self.microbatch_size = _validate_microbatch_size(
            microbatch_size, loader_batch_size
        )
        self.loader_batch_size = loader_batch_size
        self.scheduler = scheduler
        self.accumulation_steps = None
        self.use_future_meteorology = getattr(
            teacher, "use_future_meteorology", True
        )

    def train_step(self, batch):
        self.teacher.train()
        global_step_before = self.global_step
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulation_steps = derive_accumulation_steps(
            _batch_size(batch), self.microbatch_size
        )
        pinball_denominator = sum(
            int(_target_valid_mask(microbatch).sum().item())
            for microbatch in iter_microbatches(batch, self.microbatch_size)
        )
        if pinball_denominator <= 0:
            raise ValueError("training pinball denominator must be positive")
        pinball_numerator = 0.0
        for microbatch in iter_microbatches(batch, self.microbatch_size):
            device_batch = _move_batch(microbatch, self.coords.device)
            prediction, _ = self.teacher(
                **_teacher_inputs(
                    device_batch, self.coords, self.neighbor_index, self.taus,
                    self.use_future_meteorology,
                )
            )
            pinball, valid_count, numerator = _pinball_parts(
                prediction, device_batch, self.taus
            )
            _check_finite(pinball, "Teacher pinball loss")
            (pinball * (valid_count / pinball_denominator)).backward()
            pinball_numerator += numerator
            del prediction, pinball, device_batch, microbatch
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        pinball_value = pinball_numerator / pinball_denominator
        zero = torch.zeros(())
        return _step_result(
            zero.new_tensor(pinball_value),
            zero.new_tensor(pinball_value),
            zero,
            0.0,
            pinball_denominator,
            global_step_before,
            self.global_step,
            pinball_numerator,
        )

    def validation_step(self, batch):
        self.teacher.eval()
        global_step = self.global_step
        numerator = 0.0
        valid_count = 0
        with torch.no_grad():
            for microbatch in iter_microbatches(batch, self.microbatch_size):
                device_batch = _move_batch(microbatch, self.coords.device)
                prediction, _ = self.teacher(
                    **_teacher_inputs(
                        device_batch, self.coords, self.neighbor_index, self.taus,
                        self.use_future_meteorology,
                    )
                )
                _, count, part_numerator = _pinball_parts(
                    prediction, device_batch, self.taus
                )
                numerator += part_numerator
                valid_count += count
                del prediction, device_batch, microbatch
        if valid_count <= 0:
            raise ValueError("validation pinball denominator must be positive")
        pinball = torch.zeros(())
        _check_finite(pinball.new_tensor(numerator / valid_count), "Teacher validation pinball loss")
        zero = pinball.new_zeros(())
        return _step_result(
            pinball.new_tensor(numerator / valid_count),
            pinball.new_tensor(numerator / valid_count),
            zero,
            0.0,
            valid_count,
            global_step,
            global_step,
            numerator,
        )


class StudentTrainer:
    def __init__(
        self,
        student,
        teacher,
        optimizer,
        relation_loss,
        objective,
        *,
        coords,
        neighbor_index,
        taus,
        tau_weights=None,
        grad_clip=1.0,
        global_step=0,
        microbatch_size=8,
        loader_batch_size=None,
        scheduler=None,
    ):
        _validate_static_inputs(coords, neighbor_index, taus, grad_clip)
        if objective.warmup_steps is None:
            raise ValueError("Relation-KD V2 objective requires explicit warmup_steps")

        teacher_ids = {id(parameter) for parameter in teacher.parameters()}
        student_ids = {id(parameter) for parameter in student.parameters()}
        if teacher_ids & student_ids:
            raise ValueError("Teacher and Student must not share parameters")
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if teacher_ids & optimizer_ids:
            raise ValueError("Student optimizer must not contain Teacher parameters")

        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.relation_loss = relation_loss
        self.objective = objective
        self.coords = coords
        self.neighbor_index = neighbor_index
        self.taus = taus
        if tau_weights is not None:
            if not isinstance(tau_weights, torch.Tensor):
                raise TypeError("tau_weights must be None or a Tensor")
            if tau_weights.shape != taus.shape:
                raise ValueError("tau_weights must match taus")
            if not torch.isfinite(tau_weights).all() or torch.any(tau_weights < 0):
                raise ValueError("tau_weights must be finite and non-negative")
            if not torch.any(tau_weights > 0):
                raise ValueError("tau_weights must contain a positive value")
            if tau_weights.device != taus.device or tau_weights.dtype != taus.dtype:
                raise ValueError("tau_weights must match taus device and dtype")
        self.tau_weights = tau_weights
        self.grad_clip = grad_clip
        self.global_step = _validate_global_step(global_step)
        self.microbatch_size = _validate_microbatch_size(
            microbatch_size, loader_batch_size
        )
        self.loader_batch_size = loader_batch_size
        self.scheduler = scheduler
        self.accumulation_steps = None
        self.use_future_meteorology = getattr(
            teacher, "use_future_meteorology", True
        )

    def train_step(self, batch):
        self.student.train()
        self.teacher.eval()
        global_step_before = self.global_step
        relation_weight = self.objective.relation_weight(
            global_step=global_step_before
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulation_steps = derive_accumulation_steps(
            _batch_size(batch), self.microbatch_size
        )
        pinball_denominator = sum(
            int(_target_valid_mask(microbatch).sum().item())
            for microbatch in iter_microbatches(batch, self.microbatch_size)
        )
        relation_denominator = sum(
            _relation_denominator(
                _relation_group_mask(microbatch, self.use_future_meteorology),
                self.student.relation_path.relation_attention.num_heads,
                self.relation_loss,
            )
            for microbatch in iter_microbatches(batch, self.microbatch_size)
        )
        if pinball_denominator <= 0:
            raise ValueError("training pinball denominator must be positive")
        relation_weight = float(relation_weight)
        pinball_numerator = 0.0
        relation_numerator = 0.0
        for microbatch in iter_microbatches(batch, self.microbatch_size):
            device_batch = _move_batch(microbatch, self.coords.device)
            with torch.no_grad():
                teacher_corr = self.teacher.relation_target_forward(
                    **_teacher_inputs(
                        device_batch, self.coords, self.neighbor_index, self.taus,
                        self.use_future_meteorology,
                    )
                )
            prediction, student_corr = self.student(
                **_history_inputs(
                    device_batch, self.coords, self.neighbor_index, self.taus
                )
            )
            pinball, valid_count, numerator = _pinball_parts(
                prediction, device_batch, self.taus, self.tau_weights
            )
            relation_args = (
                student_corr,
                teacher_corr.detach(),
                self.student.relation_path.relation_attention.relation_gain,
                self.teacher.relation_path.relation_attention.relation_gain.detach(),
                _relation_group_mask(device_batch, self.use_future_meteorology),
            )
            rel_den_tensor = None
            if hasattr(self.relation_loss, "forward_parts"):
                rel_num, rel_den_tensor, relation = self.relation_loss.forward_parts(
                    *relation_args
                )
                rel_den = int(rel_den_tensor.item())
            else:
                relation = self.relation_loss(*relation_args)
                rel_den = _relation_denominator(
                    _relation_group_mask(microbatch, self.use_future_meteorology),
                    self.student.relation_path.relation_attention.num_heads,
                    self.relation_loss,
                )
                rel_num = relation.detach() * rel_den
            total = pinball * (valid_count / pinball_denominator)
            if relation_denominator > 0:
                total = total + relation_weight * relation * (rel_den / relation_denominator)
            _check_finite(pinball, "Student pinball loss")
            _check_finite(relation, "Student relation loss")
            _check_finite(total, "Student total loss")
            total.backward()
            pinball_numerator += numerator
            relation_numerator += float(rel_num.detach().item())
            del prediction, student_corr, teacher_corr, pinball, relation, total
            del rel_num, rel_den_tensor, relation_args, device_batch, microbatch
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        pinball_value = pinball_numerator / pinball_denominator
        relation_value = (
            relation_numerator / relation_denominator
            if relation_denominator > 0
            else 0.0
        )
        total_value = pinball_value + relation_weight * relation_value
        scalar = torch.zeros(())
        return _step_result(
            scalar.new_tensor(total_value),
            scalar.new_tensor(pinball_value),
            scalar.new_tensor(relation_value),
            relation_weight,
            pinball_denominator,
            global_step_before,
            self.global_step,
            pinball_numerator,
            relation_numerator,
            relation_denominator,
        )

    def validation_step(self, batch):
        self.student.eval()
        global_step = self.global_step
        numerator = 0.0
        valid_count = 0
        with torch.no_grad():
            for microbatch in iter_microbatches(batch, self.microbatch_size):
                device_batch = _move_batch(microbatch, self.coords.device)
                prediction, _ = self.student(
                    **_history_inputs(
                        device_batch, self.coords, self.neighbor_index, self.taus
                    )
                )
                _, count, part_numerator = _pinball_parts(
                    prediction, device_batch, self.taus
                )
                numerator += part_numerator
                valid_count += count
                del prediction, device_batch, microbatch
        if valid_count <= 0:
            raise ValueError("validation pinball denominator must be positive")
        pinball = torch.zeros(())
        _check_finite(pinball.new_tensor(numerator / valid_count), "Student validation pinball loss")
        zero = pinball.new_zeros(())
        return _step_result(
            pinball.new_tensor(numerator / valid_count),
            pinball.new_tensor(numerator / valid_count),
            zero,
            0.0,
            valid_count,
            global_step,
            global_step,
            numerator,
        )
