# Created by Junghwan (https://github.com/shaun0927)
# License: BSD-3-Clause

import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support

from pypots.nn.functional import calc_binary_classification_metrics, calc_precision_recall_f1


@pytest.mark.parametrize("pos_label", [0, 1, 2])
def test_perfect_multiclass_predictions_score_one_for_each_positive_class(pos_label: int) -> None:
    targets = np.array([0, 1, 2])
    probabilities = np.eye(3) * 0.8 + 0.2 / 3

    result = calc_binary_classification_metrics(probabilities, targets, pos_label)

    assert (result["precision"], result["recall"], result["f1"]) == (1.0, 1.0, 1.0)
    np.testing.assert_array_equal(result["predictions"], targets)
    assert result["accuracy"] == 1.0


@pytest.mark.parametrize("pos_label", [0, 1, 2])
def test_multiclass_metrics_match_one_vs_rest_scores(pos_label: int) -> None:
    targets = np.array([0, 0, 1, 1, 2, 2])
    categories = np.array([0, 1, 1, 2, 2, 2])
    probabilities = np.eye(3)[categories] * 0.8 + 0.2 / 3
    expected = precision_recall_fscore_support(targets == pos_label, categories == pos_label, average="binary")[:3]

    result = calc_binary_classification_metrics(probabilities, targets, pos_label)

    np.testing.assert_allclose([result["precision"], result["recall"], result["f1"]], expected)
    np.testing.assert_array_equal(result["predictions"], categories)
    assert result["accuracy"] == pytest.approx(4 / 6)


@pytest.mark.parametrize("shape", [(4,), (4, 1), (4, 2)])
def test_default_positive_label_preserves_binary_scores(shape: tuple) -> None:
    targets = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.2, 0.3])
    if shape == (4, 2):
        probabilities = np.column_stack([1 - probabilities, probabilities])
    else:
        probabilities = probabilities.reshape(shape)

    result = calc_binary_classification_metrics(probabilities, targets)

    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["accuracy"] == 0.75


@pytest.mark.parametrize("pos_label", [-1, 2, 7])
def test_precision_recall_f1_selects_a_label_not_an_array_position(pos_label: int) -> None:
    targets = np.array([pos_label, pos_label, 0, 0])
    categories = np.array([pos_label, 0, 0, 0])

    result = calc_precision_recall_f1(categories, targets, pos_label)

    np.testing.assert_allclose(result, [1.0, 0.5, 2 / 3])


def test_precision_recall_f1_handles_a_single_present_positive_class() -> None:
    targets = np.ones(3, dtype=int)

    result = calc_precision_recall_f1(targets.copy(), targets)

    assert result == (1.0, 1.0, 1.0)


def test_precision_recall_f1_preserves_undefined_metric_warning() -> None:
    targets = np.zeros(3, dtype=int)
    categories = targets.copy()

    with pytest.warns(UndefinedMetricWarning):
        result = calc_precision_recall_f1(categories, targets)

    assert result == (0.0, 0.0, 0.0)
