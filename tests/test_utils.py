import numpy as np

from src.utils import model_evaluate, validate_schema
import pandas as pd
import pytest


class _ConstantModel:
    """A stub 'model' that always predicts the mean of y_train, for a deterministic R2 check."""

    def fit(self, x_train, y_train):
        self._prediction = float(np.mean(y_train))
        return self

    def predict(self, x):
        return np.full(len(x), self._prediction)


def test_model_evaluate_returns_r2_per_model():
    x_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([1, 2, 3, 4])
    x_test = np.array([[5], [6]])
    y_test = np.array([5, 6])

    report = model_evaluate(x_train, y_train, x_test, y_test, {"constant": _ConstantModel()})

    assert "constant" in report
    assert isinstance(report["constant"], float)


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame({"a": [1]})
    schema = {"columns": {"a": "int64", "b": "int64"}}

    with pytest.raises(Exception):
        validate_schema(df, schema)


def test_validate_schema_passes_when_columns_match():
    df = pd.DataFrame({"a": [1], "b": [2]})
    schema = {"columns": {"a": "int64", "b": "int64"}}

    validate_schema(df, schema)
