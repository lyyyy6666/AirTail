"""Focused checks for the published AirTail model and data contracts."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from airtail.data.protocol import (
    FiniteStandardScaler, chronological_split, window_target_anchors,
)
from airtail.evaluation.paper_metrics import PaperMetrics
from airtail.models.pmg_rd.builder import build_airtail_student, build_relation_loss
from airtail.models.sa_evt import UnifiedQuantileHead
from airtail.models.temporal_stem import CausalMultiscaleTemporalStem


ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "configs/student.yaml").read_text(encoding="utf-8"))


class AirTailContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scaler_and_validation_boundary(self):
        splits = chronological_split(100, {"train": .7, "validation": .1, "test": .2})
        values = np.arange(100, dtype=np.float32).reshape(100, 1, 1)
        mask = np.ones_like(values, dtype=bool)
        fit = FiniteStandardScaler.fit(values, mask, splits["train"])
        values[70:] = 1e6
        second = FiniteStandardScaler.fit(values, mask, splits["train"])
        np.testing.assert_array_equal(fit.mean, second.mean)
        np.testing.assert_array_equal(fit.std, second.std)
        self.assertEqual(float(fit.mean.item()), 34.5)
        np.testing.assert_array_equal(window_target_anchors(splits["validation"], 24, 3), np.arange(70, 78))

    def test_temporal_prefix(self):
        torch.manual_seed(3)
        stem = CausalMultiscaleTemporalStem(8, [1, 3, 7, 24])
        x = torch.randn(2, 24, 4, 8, requires_grad=True)
        y = x.detach().clone()
        y[:, 16:] += 100
        output = stem(x)
        torch.testing.assert_close(output[:, :16], stem(y)[:, :16], rtol=0, atol=0)
        output[:, :16].square().mean().backward()
        self.assertEqual(torch.count_nonzero(x.grad[:, 16:]).item(), 0)

    def test_distillation_query_rows(self):
        loss = build_relation_loss(config())
        self.assertEqual(loss.query_rows, tuple(range(4, 12)))

    def test_sa_evt_one_sided_jet(self):
        torch.manual_seed(7)
        head = UnifiedQuantileHead(32, config()["head"]).double()
        states = torch.randn(1, 1, 1, 3, 32, dtype=torch.float64)
        median, coefficients = head.bulk.parameters_from_states(states)
        left_jet = head.bulk.endpoint_jet(median, coefficients)
        threshold, slope, curvature = left_jet
        # The spline and tail are differentiated just inside their own domains.
        for offset in [-1e-9, 1e-9]:
            tau = torch.tensor([head.tau_u + offset], dtype=torch.float64, requires_grad=True)
            value = head(states, tau).quantiles
            first = torch.autograd.grad(value.sum(), tau, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), tau)[0]
            for actual, expected in zip((value, first, second), (threshold, slope, curvature)):
                torch.testing.assert_close(actual.squeeze(), expected.squeeze(), rtol=2e-6, atol=2e-6)
        taus = torch.tensor([.5, .75, .9, .925, .95, .975, .99, .999], dtype=torch.float64)
        q = head(states, taus).quantiles
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q.diff(dim=-1) >= 0).all())

    def test_student_api_and_strict_weights(self):
        model = config()
        model.update(num_nodes=3, horizon=2)
        model["ssm"]["backend"] = "reference"
        data = json.loads((ROOT / "configs/datasets/bthsa.json").read_text())
        data["protocol"]["window"].update(prediction_steps=2, prediction_hours=2)
        student = build_airtail_student(model, data).eval()
        restored = build_airtail_student(model, data).eval()
        restored.load_state_dict(student.state_dict(), strict=True)
        inputs = dict(
            history_values=torch.zeros(1, 24, 3, 1),
            history_mask=torch.ones(1, 24, 3, 1, dtype=torch.bool),
            history_met=torch.zeros(1, 24, 3, 9),
            history_wind_uv=torch.ones(1, 24, 3, 2),
            history_time_features=torch.zeros(1, 24, 2),
            coords=torch.tensor([[0., 0.], [0., 1.], [1., 0.]]),
            neighbor_index=torch.tensor([[1, 2], [0, 2], [0, 1]]),
            taus=torch.tensor(model["taus"]["fixed"]),
        )
        with torch.no_grad():
            output, relation = student(**inputs)
            prediction = restored.deploy_forward(**inputs).quantiles
        self.assertEqual(output.quantiles.shape, (1, 2, 3, 7))
        self.assertEqual(relation.shape, (1, 2, 3, 4, 12, 12))
        torch.testing.assert_close(prediction, output.quantiles, rtol=0, atol=0)
        with self.assertRaises(TypeError):
            restored.deploy_forward(**inputs, future_met=torch.ones(1))

    def test_manuscript_metrics_in_physical_units(self):
        metric = PaperMetrics([0.5, 0.99], SimpleNamespace(mean=[4.], std=[2.]))
        y = torch.tensor([[[[0.], [2.], [100.]]]])
        q = torch.tensor([[[[-1., 1.], [-1., 1.], [0., 0.]]]])
        mask = torch.tensor([[[[True], [True], [False]]]])
        metric.update(q, y, mask)
        result = metric.compute()
        self.assertEqual(result["valid_count"], 2)
        self.assertAlmostEqual(result["picp"], 0.5)
        self.assertAlmostEqual(result["qis"], 104.0)
        self.assertAlmostEqual(result["q_risk_by_tau"]["0.5"], 1 / 1.5)
        self.assertEqual(result["by_horizon_step"]["1"]["qis"], result["qis"])
