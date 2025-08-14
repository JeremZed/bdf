import pytest
import pandas as pd
from bdf.operations.outliers import Outlier

def test_iqr_no_outliers():
    """
    Tests the case where there are no outliers in the data.
    """
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 11, 12, 13, 14]
    })
    features = ["feature1", "feature2"]

    outliers = Outlier.iqr(df, features)
    expected = pd.DataFrame(False, index=df.index, columns=features)

    pd.testing.assert_frame_equal(outliers, expected)


def test_iqr_with_outliers():
    """
    Tests the case where there are outliers in the data.
    """
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 100, 5],
        "feature2": [10, 11, 200, 13, 14]
    })
    features = ["feature1", "feature2"]

    outliers = Outlier.iqr(df, features)
    expected = pd.DataFrame({
        "feature1": [False, False, False, True, False],
        "feature2": [False, False, True, False, False]
    })

    pd.testing.assert_frame_equal(outliers, expected)


def test_iqr_empty_dataframe():
    """
    Tests the case where the DataFrame is empty.
    """
    df = pd.DataFrame(columns=["feature1", "feature2"])
    features = ["feature1", "feature2"]

    with pytest.raises(ValueError, match="The dataset must be populated."):
        outliers = Outlier.iqr(df, features)
        expected = pd.DataFrame(columns=features, index=df.index)



def test_iqr_partial_outliers():
    """
    Tests the case where some columns contain outliers but not all.
    """
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 100, 12, 13, 14]
    })
    features = ["feature1", "feature2"]

    outliers = Outlier.iqr(df, features)
    expected = pd.DataFrame({
        "feature1": [False, False, False, False, False],
        "feature2": [False, True, False, False, False]
    })

    pd.testing.assert_frame_equal(outliers, expected)


def test_iqr_different_threshold():
    """
    Tests the case where a custom threshold is used to identify outliers.
    """
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 100, 5],
        "feature2": [10, 11, 200, 13, 14]
    })
    features = ["feature1", "feature2"]

    # Stricter threshold
    outliers = Outlier.iqr(df, features, threshold=1.0)
    expected = pd.DataFrame({
        "feature1": [False, False, False, True, False],
        "feature2": [False, False, True, False, False]
    })

    pd.testing.assert_frame_equal(outliers, expected)


def test_iqr_with_missing_values():
    """
    Tests the case where missing values are present in the DataFrame.
    """
    df = pd.DataFrame({
        "feature1": [1, 2, None, 100, 5],
        "feature2": [10, 11, 200, None, 14]
    })
    features = ["feature1", "feature2"]

    outliers = Outlier.iqr(df, features)
    expected = pd.DataFrame({
        "feature1": [False, False, False, True, False],
        "feature2": [False, False, True, False, False]
    })

    pd.testing.assert_frame_equal(outliers, expected)

def test_zscore_valid_input():
    # Prepare data
    df = pd.DataFrame({
        'feature1': [10, 12, 13, 500, 11],
        'feature2': [15, 14, 500, 15, 13]
    })
    features = ['feature1', 'feature2']
    threshold = 1

    # Call the function
    outliers, z_scores = Outlier.zscore(df, features, threshold)

    # Check the size of the results
    assert outliers.shape == df.shape
    assert z_scores.shape == df.shape

    # Check specific values
    assert outliers['feature1'][3]
    assert not outliers['feature1'][0]

    assert outliers['feature2'][2]
    assert not outliers['feature2'][0]

def test_zscore_empty_dataframe():
    df = pd.DataFrame()
    features = ['feature1', 'feature2']

    with pytest.raises(ValueError, match="The dataset must be populated."):
        Outlier.zscore(df, features)

def test_zscore_non_numeric_features():
    df = pd.DataFrame({
        'feature1': [10, 12, 13, 500, 11],
        'feature2': ['a', 'b', 'c', 'd', 'e']
    })
    features = ['feature1', 'feature2']

    with pytest.raises(ValueError, match="The columns are not all of a numerical type."):
        Outlier.zscore(df, features)

def test_zscore_custom_threshold():
    df = pd.DataFrame({
        'feature1': [10, 12, 13, 50, 11],
        'feature2': [15, 14, 50, 15, 13]
    })
    features = ['feature1', 'feature2']
    threshold = 0.5

    outliers, z_scores = Outlier.zscore(df, features, threshold)

    assert outliers['feature1'][3]
    assert outliers['feature2'][2]

def test_zscore_no_outliers():
    df = pd.DataFrame({
        'feature1': [10, 11, 12, 13, 14],
        'feature2': [15, 16, 17, 18, 19]
    })
    features = ['feature1', 'feature2']
    threshold = 3

    outliers, z_scores = Outlier.zscore(df, features, threshold)

    assert not outliers.any().any()