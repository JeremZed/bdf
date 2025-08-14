from bdf.core.dataset import Dataset
from bdf.utils import Tools
from bdf.operations.visualization import Viz

import pytest
import pandas as pd
import numpy as np
import os

from unittest.mock import patch, MagicMock

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a temporary CSV file for tests."""
    data = "col1,col2,col3\n1,2,3\n4,5,6"
    filepath = tmp_path / "sample.csv"
    filepath.write_text(data)
    return filepath

@pytest.fixture
def sample_json(tmp_path):
    """Creates a temporary JSON file for tests."""
    data = [{"col1": 1, "col2": 2}, {"col1": 4, "col2": 5}]
    filepath = tmp_path / "sample.json"
    filepath.write_text(pd.DataFrame(data).to_json(orient="records"))
    return filepath

@pytest.fixture
def sample_excel(tmp_path):
    """Creates a temporary Excel file for tests."""
    data = pd.DataFrame({"col1": [1, 4], "col2": [2, 5]})
    filepath = tmp_path / "sample.xlsx"
    data.to_excel(filepath, index=False)
    return filepath

def test_load_from_dataframe():
    """Tests loading from a DataFrame."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data)
    assert dataset.df.equals(data)

def test_load_from_numpy_array():
    """Tests loading from a numpy array."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    dataset = Dataset(data)
    assert dataset.df.shape == (2, 3)
    assert (dataset.df.values == data).all()

def test_load_from_list():
    """Tests loading from a list."""
    data = [[1, 2, 3], [4, 5, 6]]
    dataset = Dataset(data)
    assert dataset.df.shape == (2, 3)
    assert dataset.df.iloc[0, 0] == 1

def test_load_from_dict():
    """Tests loading from a dictionary."""
    data = {"col1": [1, 4], "col2": [2, 5]}
    dataset = Dataset(data)
    assert dataset.df.shape == (2, 2)
    assert list(dataset.df.columns) == ["col1", "col2"]

def test_load_from_csv(sample_csv):
    """Tests loading from a CSV file."""
    dataset = Dataset(str(sample_csv))
    assert dataset.df.shape == (2, 3)
    assert list(dataset.df.columns) == ["col1", "col2", "col3"]

def test_load_from_json(sample_json):
    """Tests loading from a JSON file."""
    dataset = Dataset(str(sample_json))
    assert dataset.df.shape == (2, 2)
    assert list(dataset.df.columns) == ["col1", "col2"]

def test_load_from_excel(sample_excel):
    """Tests loading from an Excel file."""
    dataset = Dataset(str(sample_excel))
    assert dataset.df.shape == (2, 2)
    assert list(dataset.df.columns) == ["col1", "col2"]

def test_file_not_found():
    """Tests the case where the file does not exist."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        Dataset("invalid_path.csv")

def test_unsupported_extension(tmp_path):
    """Tests loading a file with an unsupported extension."""
    unsupported_file = tmp_path / "data.txt"
    unsupported_file.write_text("data")
    with pytest.raises(ValueError, match="Unsupported extension"):
        Dataset(str(unsupported_file))

def test_reset_method():
    """Tests the reset method to reset the state."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data)
    dataset.reset()
    assert dataset.df is None

def test_invalid_data_type():
    """Tests an unsupported data type."""
    with pytest.raises(ValueError, match="Invalid data type"):
        Dataset(12345)

def test_logging_verbose(capfd):
    """Tests that logs are displayed when verbose is enabled."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data, options={"verbose": 1})
    captured = capfd.readouterr()
    assert "Resetting dataset" in captured.out
    assert "Data loaded successfully" in captured.out

def test_logging_not_verbose(capfd):
    """Tests that logs are not displayed when verbose is disabled."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data, options={"verbose": 0})
    captured = capfd.readouterr()
    assert captured.out == ""


@pytest.fixture
def sample_dataset():
    """Fixture to create a sample dataset."""
    data = {
        "col1": [1, 2, 3, 4, 5],
        "col2": [5, 4, 3, 2, 1],
        "col3": [np.nan, 2, 3, np.nan, 5]
    }
    df = pd.DataFrame(data)
    return Dataset(df)

def test_head(sample_dataset):
    """Tests the head() function."""
    result = sample_dataset.head(3)
    assert result.shape == (3, 3)
    assert result["col1"].iloc[0] == 1

def test_tail(sample_dataset):
    """Tests the tail() function."""
    result = sample_dataset.tail(2)
    assert result.shape == (2, 3)
    assert result["col1"].iloc[0] == 4

def test_info(sample_dataset):
    """Tests the info() function."""
    result = sample_dataset.info()
    assert result is None

def test_describe(sample_dataset):
    """Tests the describe() function."""
    result = sample_dataset.describe()
    assert "col1" in result.columns
    assert result["col1"]["mean"] == 3

def test_shape(sample_dataset):
    """Tests the shape() function."""
    result = sample_dataset.shape()
    assert result == (5, 3)

def test_missing_values(sample_dataset):
    """Tests the missing_values() function."""
    result = sample_dataset.missing_values()
    assert result.shape == (4, 2)
    assert result["ratio"].iloc[0] == 2.0

def test_drop_missing_values(sample_dataset):
    """Tests the drop_missing_values() function."""
    result = sample_dataset.drop_missing_values()
    assert result.df.shape == (3, 3)
    assert result.df.isna().sum().sum() == 0

def test_fill_missing(sample_dataset):
    """Tests the fill_missing() function with 'mean' strategy."""
    result = sample_dataset.fill_missing(strategy='mean')
    assert result.df["col3"].isna().sum() == 0
    assert result.df["col3"].iloc[0] == 3.3333333333333335

def test_duplicated_values(sample_dataset):
    """Tests the duplicated_values() function."""
    result = sample_dataset.duplicated_values()
    assert result == 0

    # Create a duplicate
    sample_dataset.df.loc[len(sample_dataset.df.index)] = [3,3,3]

    result = sample_dataset.duplicated_values()
    assert result == 2


def test_drop_duplicated_values(sample_dataset):
    """Tests the drop_duplicated_values() function."""
    sample_dataset.df.loc[len(sample_dataset.df.index)] = [3,3,3]
    sample_dataset.drop_duplicated_values()

    assert sample_dataset.df.shape == (5, 3)

def test_dtypes(sample_dataset):
    """Tests the dtypes() function."""
    result = sample_dataset.dtypes()
    assert result["col1"] == np.int64
    assert result["col2"] == np.int64
    assert result["col3"] == float

    result_count = sample_dataset.dtypes(mode="count")

    assert result_count.iloc[0] == 2

def test_convert_dtypes(sample_dataset):
    """Tests the convert_dtypes() function."""
    result = sample_dataset.convert_dtypes({"col1": np.float64})
    assert result.df["col1"].dtype == np.float64

def test_normalize(sample_dataset):
    """Tests the normalize() function."""
    result = sample_dataset.normalize(columns=["col1"])
    assert result.df["col1"].min() == 0
    assert result.df["col1"].max() == 1

def test_standardize(sample_dataset):
    """Tests the standardize() function."""
    result = sample_dataset.standardize(columns=["col1"])
    assert result.df["col1"].mean() == pytest.approx(0, 1e-6)
    assert result.df["col1"].std() == pytest.approx(1, 1e-6)

def test_value_counts(sample_dataset):
    """Tests the value_counts() function."""
    result = sample_dataset.value_counts("col1")
    assert result[1] == 1
    assert result[5] == 1

def test_correlations(sample_dataset):
    """Tests the correlations() function."""
    result = sample_dataset.correlations()
    assert result.shape == (3, 3)
    assert result["col1"]["col2"] == -1

def test_top_values(sample_dataset):
    """Tests the top_values() function."""
    result = sample_dataset.top_values(n=2)
    assert result.shape == (2, 6)
    assert result["col1"]["value"].iloc[0] == 1

def test_filter_rows(sample_dataset):
    """Tests the filter_rows() function."""
    result = sample_dataset.filter_rows("col1 > 2")
    assert result.shape == (3, 3)
    assert result["col1"].iloc[0] == 3

def test_add_column(sample_dataset):
    """Tests the add_column() function."""
    result = sample_dataset.add_column("col4", [10, 20, 30, 40, 50])
    assert "col4" in result.df.columns
    assert result.df["col4"].iloc[0] == 10


@pytest.fixture
def setup_method():
    """Fixture to create a sample dataset."""

    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
        "feature2": [15, 14, 500, 15, 13],
        "feature3": [1.2, 2.5, 3.1, 4.8, 5.5],
        "non_numeric": ["A", "B", "C", "D", "E"]
    })
    return Dataset(df)


def test_outliers_iqr(setup_method):
    """Basic test with the IQR method."""
    result = setup_method.outliers(columns=["feature1", "feature3"], method="IQR")
    assert isinstance(result, pd.DataFrame)
    assert "count" in result.columns
    assert "ratio" in result.columns
    assert result.loc["feature1", "count"] == 1
    assert result.loc["feature3", "count"] == 0

def test_outliers_with_columns(setup_method):
    """Test by specifying a list of columns."""
    result = setup_method.outliers(columns=["feature1", "feature3"], method="IQR")
    assert "feature2" not in result.index
    assert "feature1" in result.index
    assert "feature3" in result.index

def test_outliers_empty_dataframe(setup_method):
    """Test with an empty DataFrame."""
    empty_df = pd.DataFrame()
    instance = Dataset(empty_df)
    with pytest.raises(ValueError, match="The dataset must be populated."):
        instance.outliers()

def test_outliers_non_numeric_columns(setup_method):
    """Test with non-numeric columns."""
    with pytest.raises(ValueError, match="The columns are not all of a numerical type."):
        setup_method.outliers(columns=["non_numeric"])

def test_outliers_invalid_method(setup_method):
    """Test with an invalid method."""
    with pytest.raises(ValueError, match="The calculation method is not supported: invalid_method"):
        setup_method.outliers(method="invalid_method")

def test_outliers_show_graph(setup_method):
    """Test with graph display enabled."""
    with patch("bdf.operations.visualization.Viz.plot_outliers") as mock_plot:
        setup_method.outliers(show_graph=True)
        mock_plot.assert_called_once()

def test_outliers_ratio_calculation(setup_method):
    """Test the calculation of the outlier ratio."""
    result = setup_method.outliers(method="IQR")
    assert result.loc["feature1", "ratio"] == 20.0
    assert result.loc["feature2", "ratio"] == 20.0
    assert result.loc["feature3", "ratio"] == 0.0
    assert result.loc["BDF_total_of_values", "ratio"] == 10

def test_outliers_custom_threshold(setup_method):
    """Test with a custom parameter for the threshold."""
    result = setup_method.outliers(method="IQR", threshold=3)
    assert result.loc["feature1", "count"] == 1
    assert result.loc["feature3", "count"] == 0