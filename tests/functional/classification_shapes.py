# Created by Junghwan (https://github.com/shaun0927)
# License: BSD-3-Clause

import numpy as np
import pytest

from pypots.nn.functional import calc_binary_classification_metrics


@pytest.mark.parametrize("shape", [(), (2, 2), (2, 1, 1)])
def test_invalid_target_dimensions_raise_value_error(shape: tuple) -> None:
    targets = np.zeros(shape)
    probabilities = np.array([0.1, 0.9])

    with pytest.raises(ValueError, match="targets.shape"):
        calc_binary_classification_metrics(probabilities, targets)


@pytest.mark.parametrize("shape", [(), (2, 0), (2, 1, 1)])
def test_invalid_prediction_dimensions_raise_value_error(shape: tuple) -> None:
    probabilities = np.zeros(shape)
    targets = np.array([0, 1])

    with pytest.raises(ValueError, match="predictions.shape"):
        calc_binary_classification_metrics(probabilities, targets)


@pytest.mark.parametrize("target_shape", [(2,), (2, 1)])
@pytest.mark.parametrize("prediction_shape", [(2,), (2, 1), (2, 2)])
def test_supported_dimensions_preserve_perfect_predictions(target_shape: tuple, prediction_shape: tuple) -> None:
    targets = np.array([0, 1]).reshape(target_shape)
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    if prediction_shape != (2, 2):
        probabilities = probabilities[:, 1].reshape(prediction_shape)

    result = calc_binary_classification_metrics(probabilities, targets)

    np.testing.assert_array_equal(result["predictions"], [0, 1])
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
