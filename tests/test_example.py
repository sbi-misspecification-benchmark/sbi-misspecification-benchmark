import pytest

def test_addition():
    """Example test: Checks if addition works correctly."""
    assert 1 + 1 == 2, "Addition result should be 2"

def test_subtraction():
    """Example test: Checks if subtraction works correctly."""
    assert 5 - 3 == 2, "Substraction result should be 2"

def test_multiplication():
    """Example test: Checks if multiplication works correctly."""
    result = 3 * 4
    assert result == 12, "Multiplication result should be 12"

def test_division():
    """Example test: Checks if division works correctly."""
    result = 10 / 2
    assert result == 5, "Division result should be 5"

def test_throw_error():
    """Example test: Checks if throw Exception works correctly."""
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0