import pytest
import pandas as pd
from bdf.outliers import Outlier

def test_iqr_no_outliers():
    """
    Teste le cas où il n'y a pas d'outliers dans les données.
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
    Teste le cas où il y a des outliers dans les données.
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
    Teste le cas où le DataFrame est vide.
    """
    df = pd.DataFrame(columns=["feature1", "feature2"])
    features = ["feature1", "feature2"]

    with pytest.raises(ValueError, match="Le dataset doit être alimenté."):
        outliers = Outlier.iqr(df, features)
        expected = pd.DataFrame(columns=features, index=df.index)



def test_iqr_partial_outliers():
    """
    Teste le cas où certaines colonnes contiennent des outliers mais pas toutes.
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
    Teste le cas où un seuil personnalisé est utilisé pour identifier les outliers.
    """
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 100, 5],
        "feature2": [10, 11, 200, 13, 14]
    })
    features = ["feature1", "feature2"]

    # Seuil plus strict
    outliers = Outlier.iqr(df, features, threshold=1.0)
    expected = pd.DataFrame({
        "feature1": [False, False, False, True, False],
        "feature2": [False, False, True, False, False]
    })

    pd.testing.assert_frame_equal(outliers, expected)


def test_iqr_with_missing_values():
    """
    Teste le cas où des valeurs manquantes sont présentes dans le DataFrame.
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
