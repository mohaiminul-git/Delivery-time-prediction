import sys

import pytest

from src.exceptions import CustomException


def test_custom_exception_includes_original_message():
    try:
        1 / 0
    except Exception as e:
        error = CustomException(e, sys)

    assert "division by zero" in str(error)


def test_custom_exception_includes_file_and_line_number():
    try:
        raise ValueError("boom")
    except Exception as e:
        error = CustomException(e, sys)

    assert __file__ in str(error) or "test_exceptions.py" in str(error)


def test_custom_exception_repr():
    try:
        raise ValueError("boom")
    except Exception as e:
        error = CustomException(e, sys)

    assert repr(error).startswith("CustomException(")
