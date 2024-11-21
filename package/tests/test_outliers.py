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

def test_zscore_valid_input():
    # Préparation des données
    df = pd.DataFrame({
        'feature1': [10, 12, 13, 500, 11],
        'feature2': [15, 14, 500, 15, 13]
    })
    features = ['feature1', 'feature2']
    threshold = 1

    # Appel de la fonction
    outliers, z_scores = Outlier.zscore(df, features, threshold)

    # Vérification de la taille des résultats
    assert outliers.shape == df.shape
    assert z_scores.shape == df.shape

    # Vérification des valeurs spécifiques
    assert outliers['feature1'][3]  # L'indice 3 doit être un outlier
    assert not outliers['feature1'][0]  # L'indice 0 ne doit pas être un outlier

    assert outliers['feature2'][2]  # L'indice 2 doit être un outlier
    assert not outliers['feature2'][0]  # L'indice 0 ne doit pas être un outlier

def test_zscore_empty_dataframe():
    df = pd.DataFrame()
    features = ['feature1', 'feature2']

    with pytest.raises(ValueError, match="Le dataset doit être alimenté."):
        Outlier.zscore(df, features)

def test_zscore_non_numeric_features():
    df = pd.DataFrame({
        'feature1': [10, 12, 13, 500, 11],
        'feature2': ['a', 'b', 'c', 'd', 'e']  # Non-numeric feature
    })
    features = ['feature1', 'feature2']

    with pytest.raises(ValueError, match="Les colonnes de sont pas toutes de type numérique."):
        Outlier.zscore(df, features)

def test_zscore_custom_threshold():
    df = pd.DataFrame({
        'feature1': [10, 12, 13, 50, 11],
        'feature2': [15, 14, 50, 15, 13]
    })
    features = ['feature1', 'feature2']
    threshold = 0.5  # Seuil plus bas pour détecter les outliers

    outliers, z_scores = Outlier.zscore(df, features, threshold)

    # Vérification des outliers avec le seuil réduit
    assert outliers['feature1'][3]  # L'indice 3 doit être un outlier
    assert outliers['feature2'][2]  # L'indice 2 doit être un outlier

def test_zscore_no_outliers():
    df = pd.DataFrame({
        'feature1': [10, 11, 12, 13, 14],
        'feature2': [15, 16, 17, 18, 19]
    })
    features = ['feature1', 'feature2']
    threshold = 3

    outliers, z_scores = Outlier.zscore(df, features, threshold)

    # Aucune colonne ne doit contenir d'outliers
    assert not outliers.any().any()