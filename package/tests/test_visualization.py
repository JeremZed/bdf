import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from matplotlib import pyplot as plt
from bdf.operations.visualization import Viz
from bdf.operations.outliers import Outlier

@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for top values."""
    data = {
        ('col1', 'value'): ['A', 'B', 'C', 'D', 'E'],
        ('col1', 'count'): [5, 10, 15, 10, 5],
        ('col2', 'value'): ['X', 'Y', 'Z', np.nan, np.nan],
        ('col2', 'count'): [20, 25, 30, 0, 0]
    }
    df = pd.DataFrame(data)
    return df

def test_plot_top_values_basic(sample_df):
    """Tests the plot_top_values method to verify that it generates charts without errors."""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=2)

def test_plot_top_values_with_different_columns(sample_df):
    """Tests to verify that the method correctly handles a different number of columns."""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1)

def test_plot_top_values_titles_and_labels(sample_df):
    """Tests to verify that titles and labels are correctly set."""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1)

    fig = plt.gcf()
    ax = fig.get_axes()[0]
    assert ax.get_title() == "Top Values for col1"
    assert ax.get_xlabel() == "Values"
    assert ax.get_ylabel() == "Counts"

def test_plot_top_values_y_axis(sample_df):
    """Tests to verify the behavior of the Y-axis based on the show_y parameter."""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1, show_y=True)

    fig = plt.gcf()
    ax = fig.get_axes()[0]
    assert len(ax.get_yticks()) > 0

    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1, show_y=False)

    fig = plt.gcf()
    ax = fig.get_axes()[0]
    assert len(ax.get_yticks()) == 0

def test_plot_top_values_subplots_layout(sample_df):
    """Tests to verify the layout of the subplots."""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=2)

    fig = plt.gcf()
    axes = fig.get_axes()
    assert len(axes) > 1

def test_plot_top_values_empty_dataframe():
    """Tests with an empty DataFrame to verify handling of empty cases."""
    empty_df = pd.DataFrame(columns=[('col1', 'value'), ('col1', 'count')])
    with pytest.raises(ValueError, match="The dataset must be populated."):
        Viz.plot_top_values(empty_df, nb_cols=1)

def test_plot_outliers_iqr_valid_input():
    """Tests with valid inputs to verify that no exception is raised."""
    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
        "feature2": [15, 14, 500, 15, 13],
    })
    outliers = pd.DataFrame({
        "feature1": [False, False, False, True, False],
        "feature2": [False, False, True, False, False],
    }, index=df.index)

    with patch("matplotlib.pyplot.show") as mock_show:
        Viz.plot_outliers(df, outliers, method=Outlier.METHOD_IQR, columns=["feature1", "feature2"])
        mock_show.assert_called_once()


def test_plot_outliers_iqr_empty_dataframe():
    """Tests with an empty DataFrame to verify that an exception is raised."""
    df = pd.DataFrame()
    outliers = pd.DataFrame()

    with pytest.raises(ValueError, match="The dataset must be populated."):
        Viz.plot_outliers(df, outliers, columns=[], method=Outlier.METHOD_IQR)


def test_plot_outliers_iqr_single_column():
    """Tests with a single column to visualize."""
    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
    })
    outliers = pd.DataFrame({
        "feature1": [False, False, False, True, False],
    }, index=df.index)

    with patch("matplotlib.pyplot.show") as mock_show:
        Viz.plot_outliers(df, outliers, columns=["feature1"], method=Outlier.METHOD_IQR)
        mock_show.assert_called_once()


def test_plot_outliers_iqr_multiple_columns():
    """Tests with multiple columns to verify the layout of the subplots."""
    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
        "feature2": [15, 14, 500, 15, 13],
        "feature3": [1, 2, 3, 4, 5],
    })
    outliers = pd.DataFrame({
        "feature1": [False, False, False, True, False],
        "feature2": [False, False, True, False, False],
        "feature3": [False, False, False, False, False],
    }, index=df.index)

    with patch("matplotlib.pyplot.show") as mock_show:
        Viz.plot_outliers(df, outliers, columns=["feature1", "feature2", "feature3"], nb_cols=2, method=Outlier.METHOD_IQR)
        mock_show.assert_called_once()

def test_plot_outliers_iqr_show_y_ticks():
    """Tests with the `show_y=True` option to verify the addition of y-axis ticks."""
    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
    })
    outliers = pd.DataFrame({
        "feature1": [False, False, False, True, False],
    }, index=df.index)

    with patch("matplotlib.pyplot.show") as mock_show:
        Viz.plot_outliers(df, outliers, columns=["feature1"], method=Outlier.METHOD_IQR, show_y=True)
        mock_show.assert_called_once()


def test_plot_zscore_empty_z_scores():
    """
    Tests if the function raises an exception for an empty DataFrame.
    """
    z_scores = pd.DataFrame()
    columns = ['feature1', 'feature2']

    with pytest.raises(ValueError, match="The dataset must be populated."):
        Viz.plot_zscore(z_scores, columns)

def test_plot_zscore_invalid_columns():
    """
    Tests if the function raises an exception when the specified columns are not in the DataFrame.
    """
    z_scores = pd.DataFrame({
        'feature1': [0.5, -0.8, 1.2, -0.3, 2.0],
        'feature2': [1.5, -1.3, 0.8, 0.2, -0.7]
    })
    columns = ['feature3']

    with pytest.raises(ValueError, match="Not all features are available."):
        Viz.plot_zscore(z_scores, columns)
