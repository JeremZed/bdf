import os
import pytest
import pandas as pd
import numpy as np

from bdf.utils import Tools

LOG_FILENAME = "bdf.log"
LOG_FILE_PATH = "test_logs"

@pytest.fixture(scope="module")
def create_log_directory():
    """Creates a directory to store logs during tests."""
    if not os.path.exists(LOG_FILE_PATH):
        os.makedirs(LOG_FILE_PATH)
    yield
    # Cleanup after tests
    if os.path.exists(LOG_FILE_PATH):
        for file in os.listdir(LOG_FILE_PATH):
            os.remove(os.path.join(LOG_FILE_PATH, file))
        os.rmdir(LOG_FILE_PATH)


def test_log_to_console(capsys):
    """Tests if the log message is displayed correctly in the console."""
    message = "Test log message"
    level = 1
    Tools.log(message, level, show=True, write=False, threshold=1)

    captured = capsys.readouterr()

    assert message in captured.out
    assert "[LOG]" in captured.out


def test_log_to_file(create_log_directory):
    """Tests if the log message is written correctly to a file."""
    message = "Test log file"
    level = 1
    pathfile = os.path.join(LOG_FILE_PATH, LOG_FILENAME)

    Tools.log(message, level, show=False, write=True, threshold=1, pathfile=pathfile)

    assert os.path.exists(pathfile)
    with open(pathfile, 'r') as f:
        content = f.read()
    assert message in content


def test_log_with_invalid_directory():
    """Tests if an exception is raised when the directory is not found."""
    message = "Test invalid directory"
    level = 1
    invalid_path = "invalid_directory/some.log"

    with pytest.raises(FileNotFoundError):
        Tools.log(message, level, show=False, write=True, threshold=1, pathfile=invalid_path)


def test_log_with_invalid_pathfile():
    """Tests if an exception is raised for an incorrect file path."""
    message = "Test invalid pathfile"
    level = 1
    invalid_pathfile = "invalid_path_file.txt"

    with pytest.raises(Exception):
        Tools.log(message, level, show=False, write=True, threshold=1, pathfile=invalid_pathfile)


def test_log_with_different_levels(capsys):
    """Tests if the log level works correctly with the threshold."""
    message = "Test log with levels"
    level_info = 1
    level_warning = 2
    threshold = 2

    Tools.log(message, level_info, show=True, write=False, threshold=threshold)
    captured = capsys.readouterr()
    assert message not in captured.out

    Tools.log(message, level_warning, show=True, write=False, threshold=threshold)
    captured = capsys.readouterr()
    assert message in captured.out


@pytest.mark.parametrize("show, expected_output", [
    (True, "Test log message"),
    (False, "")
])
def test_log_show_parameter(capsys, show, expected_output):
    """Tests the 'show' parameter to display or not display the log in the console."""
    message = "Test log message"
    level = 1
    Tools.log(message, level, show=show, write=False, threshold=1)
    captured = capsys.readouterr()

    assert expected_output in captured.out

def test_default_generation():
    """Tests default generation with default parameters."""
    result = Tools.random_id()
    assert len(result) == 12
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz0123456789-_!?@$*."
    assert all(char in valid_chars for char in result)

def test_custom_length():
    """Tests with a custom length."""
    length = 20
    result = Tools.random_id(length=length)
    assert len(result) == length

def test_custom_special_characters():
    """Tests with a custom set of special characters."""
    chars_special = "!@#"
    result = Tools.random_id(chars_special=chars_special)
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz0123456789" + chars_special
    assert all(char in valid_chars for char in result)

def test_excludes_characters():
    """Tests with excluded characters."""
    excludes_chars = ['A', '1', '!', 'z']
    result = Tools.random_id(excludes_chars=excludes_chars)
    assert not any(char in excludes_chars for char in result)

def test_pattern_generation():
    """Tests with a specific pattern."""
    pattern = ["%S", "%s", "%d", "%x", "X", "%S"]
    result = Tools.random_id(pattern=pattern)
    assert len(result) == len(pattern)
    assert result[0].isupper()
    assert result[1].islower()
    assert result[2].isdigit()
    assert result[3] in "-_!?@$*."
    assert result[4] == "X"
    assert result[5].isupper()

def test_prevent_duplicate_ids():
    """Tests to avoid duplicates in the list of identifiers."""
    existing_ids = ["ABC123", "XYZ789"]
    new_id = Tools.random_id(length=6, uids=existing_ids)
    assert new_id not in existing_ids

def test_invalid_length():
    """Tests with an invalid length."""
    with pytest.raises(ValueError, match="the length parameter must be a positive integer."):
        Tools.random_id(length=-5)

def test_invalid_special_characters():
    """Tests with invalid special characters."""
    with pytest.raises(ValueError, match="the chars_special parameter must be a string."):
        Tools.random_id(chars_special=123)

def test_invalid_pattern_type():
    """Tests with an invalid pattern (not a list)."""
    with pytest.raises(ValueError, match="The pattern parameter must be a list containing at least one element."):
        Tools.random_id(pattern="invalid_pattern")

def test_empty_pattern():
    """Tests with an empty pattern."""
    with pytest.raises(ValueError, match="The pattern parameter must be a list containing at least one element."):
        Tools.random_id(pattern=[])

def test_all_characters_excluded():
    """Tests where all characters are excluded."""
    excludes_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz0123456789-_!?@$*.")
    with pytest.raises(ValueError, match="All possible characters are excluded. Cannot generate an identifier."):
        Tools.random_id(excludes_chars=excludes_chars)

def test_no_valid_items_in_get_random_element():
    """Tests with an empty list or impossible exclusions for get_random_element."""
    with pytest.raises(ValueError, match="No element is available for selection."):
        Tools.get_random_element(["A", "B"], excludes=["A", "B"])

def test_recursive_id_generation():
    """Tests recursive generation with initial duplicates."""
    existing_ids = ["ABCDEF" for _ in range(100)]
    new_id = Tools.random_id(length=6, uids=existing_ids)
    assert new_id not in existing_ids

def test_all_numeric_columns():
    """Tests with a DataFrame containing only numeric columns."""
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4.5, 5.6, 6.7],
        "col3": [0, -1, 2]
    })
    assert Tools.is_all_numeric(df) is True

def test_mixed_columns():
    """Tests with a DataFrame containing numeric and non-numeric columns."""
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "col3": [4.5, 5.6, 6.7]
    })
    assert Tools.is_all_numeric(df) is False

def test_all_non_numeric_columns():
    """Tests with a DataFrame containing only non-numeric columns."""
    df = pd.DataFrame({
        "col1": ["x", "y", "z"],
        "col2": ["a", "b", "c"]
    })
    assert Tools.is_all_numeric(df) is False

def test_empty_dataframe():
    """Tests with an empty DataFrame."""
    df = pd.DataFrame()
    assert Tools.is_all_numeric(df) is True

def test_no_columns_dataframe():
    """Tests with a DataFrame without columns but with rows."""
    df = pd.DataFrame(index=[0, 1, 2])
    assert Tools.is_all_numeric(df) is True

def test_single_numeric_column():
    """Tests with a DataFrame containing a single numeric column."""
    df = pd.DataFrame({
        "col1": [1.1, 2.2, 3.3]
    })
    assert Tools.is_all_numeric(df) is True

def test_single_non_numeric_column():
    """Tests with a DataFrame containing a single non-numeric column."""
    df = pd.DataFrame({
        "col1": ["a", "b", "c"]
    })
    assert Tools.is_all_numeric(df) is False

def test_column_with_nan_values():
    """Tests with a DataFrame containing NaN values in a numeric column."""
    df = pd.DataFrame({
        "col1": [1, np.nan, 3],
        "col2": [4.5, 5.6, np.nan]
    })
    assert Tools.is_all_numeric(df) is True

def test_columns_with_different_dtypes():
    """Tests with a DataFrame containing columns of different types."""
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4.5, 5.6, 6.7],
        "col3": ["a", "b", "c"]
    })
    assert Tools.is_all_numeric(df) is False