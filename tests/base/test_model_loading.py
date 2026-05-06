"""
Focused tests for BaseModel.load safety defaults.
"""

import inspect
import tempfile
import unittest
from unittest.mock import patch

import torch

from pypots.base import BaseModel


class DummyModel(BaseModel):
    def __init__(self):
        super().__init__(device=torch.device("cpu"), verbose=False)
        self.model = torch.nn.Linear(2, 2)

    def fit(self, train_set, val_set=None, file_type: str = "hdf5"):
        raise NotImplementedError

    def predict(self, test_set, file_type: str = "hdf5"):
        raise NotImplementedError


class TestBaseModelLoad(unittest.TestCase):
    def test_load_uses_safe_torch_load_defaults_when_supported(self):
        model = DummyModel()
        checkpoint = {"model_state_dict": model.model.state_dict()}
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters

        with tempfile.NamedTemporaryFile(suffix=".pypots") as tmp:
            torch.save(checkpoint, tmp.name)

            with patch("pypots.base.torch.load", wraps=torch.load) as mocked_torch_load:
                model.load(tmp.name)

            load_kwargs = mocked_torch_load.call_args.kwargs
            if supports_weights_only:
                self.assertIs(load_kwargs["weights_only"], False) # TODO: Consider setting to True in the future for added safety, but need to verify it doesn't break any existing models first
            else:
                self.assertNotIn("weights_only", load_kwargs)

    def test_load_restores_saved_state_dict(self):
        source_model = DummyModel()
        source_model.model.bias.data.fill_(3.14)

        target_model = DummyModel()
        target_model.model.bias.data.zero_()

        with tempfile.NamedTemporaryFile(suffix=".pypots") as tmp:
            torch.save({"model_state_dict": source_model.model.state_dict()}, tmp.name)
            target_model.load(tmp.name)

        self.assertTrue(torch.allclose(target_model.model.bias, source_model.model.bias))


if __name__ == "__main__":
    unittest.main()
