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

def test_string():
    """Example test: Checks if string conotation works correctly."""
    string1 = "Hello"
    string2 = "World"
    result = string1 + " " + string2
    assert result == "Hello World", "String concatenation result should be 'Hello World'"

def test_list():
    """Example test: Checks if list appending works correctly."""
    my_list = [1, 2, 3]
    my_list.append(4)
    assert my_list == [1, 2, 3, 4], "List should have 4 appended to it"

def test_hashmap():
    """Example test: Checks if Hashmap value retrieval works correctly."""
    my_dict = {"a": 1, "b": 2}
    assert my_dict["a"] == 1, "Key 'a' should return value 1"

def test_mod():
    """Example test: Checks if modulo works correctly."""
    number = 4
    assert number % 2 == 0, "Number should be even"

def test_set():
    """Example test: Checks if set operations works correctly."""
    my_set = {1, 2, 3}
    assert 2 in my_set, "Set should contain the number 2"