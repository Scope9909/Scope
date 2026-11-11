"""Focused correctness checks for the Scope factorized predictor."""

import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.Scope import (  # noqa: E402
    factorized_richer_predictor,
    richer_predictor,
)


class FactorizedLitePredictorTest(unittest.TestCase):
    def test_full_rank_migration_is_functionally_equivalent(self):
        torch.manual_seed(7)
        input_length = 3
        hidden_dim = 4
        output_width = 8
        horizon = 3
        input_features = input_length * hidden_dim

        teacher = richer_predictor(
            num_nodes=5,
            input_length=input_length,
            predict_length=horizon,
            hid_dim=hidden_dim,
            pre_dim=1,
            num_of_filters=output_width,
            use_future_time_decoder=False,
            use_step_embedding=True,
        ).eval()
        student = factorized_richer_predictor(
            num_nodes=5,
            input_length=input_length,
            predict_length=horizon,
            hid_dim=hidden_dim,
            pre_dim=1,
            num_of_filters=output_width,
            factor_rank=min(input_features, output_width),
        ).eval()
        parent_state = {
            'time_wise_predictor.' + key: value
            for key, value in teacher.state_dict().items()
        }
        retained = student.initialize_from_richer_state_dict(
            parent_state, prefix='time_wise_predictor.'
        )
        sample = torch.randn(2, input_length, 5, hidden_dim)
        with torch.inference_mode():
            teacher_output = teacher(sample)
            student_output = student(sample)

        self.assertAlmostEqual(retained, 1.0, places=5)
        torch.testing.assert_close(
            student_output, teacher_output, rtol=2e-5, atol=2e-5
        )

    def test_r256_parameter_count(self):
        model = factorized_richer_predictor(
            num_nodes=307,
            input_length=12,
            predict_length=12,
            hid_dim=64,
            pre_dim=1,
            num_of_filters=1024,
            factor_rank=256,
        )
        actual = sum(parameter.numel() for parameter in model.parameters())
        expected = (
            768 * 256
            + 256 * 1024 + 1024
            + 12 * (1024 + 1)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual, 472_076)

    def test_activation_aware_full_rank_is_equivalent(self):
        torch.manual_seed(11)
        input_length = 2
        hidden_dim = 3
        output_width = 5
        input_features = input_length * hidden_dim
        teacher = richer_predictor(
            num_nodes=4,
            input_length=input_length,
            predict_length=2,
            hid_dim=hidden_dim,
            pre_dim=1,
            num_of_filters=output_width,
            use_future_time_decoder=False,
            use_step_embedding=True,
        ).eval()
        student = factorized_richer_predictor(
            num_nodes=4,
            input_length=input_length,
            predict_length=2,
            hid_dim=hidden_dim,
            pre_dim=1,
            num_of_filters=output_width,
            factor_rank=min(input_features, output_width),
        ).eval()
        random_matrix = torch.randn(input_features, input_features)
        second_moment = random_matrix @ random_matrix.T
        state = {
            'time_wise_predictor.' + key: value
            for key, value in teacher.state_dict().items()
        }
        student.initialize_from_richer_state_dict(
            state,
            prefix='time_wise_predictor.',
            activation_second_moment=second_moment,
            activation_ridge=1e-6,
        )
        sample = torch.randn(3, input_length, 4, hidden_dim)
        with torch.inference_mode():
            expected = teacher(sample)
            actual = student(sample)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
