import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from matplotlib import pyplot as plt
from bdf.visualization import Viz

@pytest.fixture
def sample_df():
    """Fixture pour créer un DataFrame d'exemple pour les top values."""
    data = {
        ('col1', 'value'): ['A', 'B', 'C', 'D', 'E'],
        ('col1', 'count'): [5, 10, 15, 10, 5],
        ('col2', 'value'): ['X', 'Y', 'Z', np.nan, np.nan],
        ('col2', 'count'): [20, 25, 30, 0, 0]
    }
    df = pd.DataFrame(data)
    return df

def test_plot_top_values_basic(sample_df):
    """Test de la méthode plot_top_values pour vérifier qu'elle génère des graphiques sans erreur."""
    # Mock plt.show pour éviter d'afficher les graphiques pendant les tests
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=2)

def test_plot_top_values_with_different_columns(sample_df):
    """Test pour vérifier que la méthode gère correctement un nombre différent de colonnes."""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1)

def test_plot_top_values_titles_and_labels(sample_df):
    """Test pour vérifier que les titres et les étiquettes sont correctement définis."""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1)

    # Vérifier si le titre et les labels sont présents
    fig = plt.gcf()  # Récupère la figure actuelle
    ax = fig.get_axes()[0]  # Récupère le premier axe
    assert ax.get_title() == "Top Values for col1"  # Vérifie que le titre est correct
    assert ax.get_xlabel() == "Values"  # Vérifie l'étiquette de l'axe X
    assert ax.get_ylabel() == "Counts"  # Vérifie l'étiquette de l'axe Y

def test_plot_top_values_y_axis(sample_df):
    """Test pour vérifier le comportement de l'axe Y en fonction du paramètre show_y."""
    # Test avec show_y=True
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1, show_y=True)

    # Vérifier que l'axe Y est défini
    fig = plt.gcf()
    ax = fig.get_axes()[0]
    assert len(ax.get_yticks()) > 0  # Vérifie que l'axe Y a des graduations

    # Test avec show_y=False
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=1, show_y=False)

    # Vérifier que l'axe Y est masqué
    fig = plt.gcf()
    ax = fig.get_axes()[0]
    assert len(ax.get_yticks()) == 0  # Vérifie que l'axe Y n'a pas de graduations

def test_plot_top_values_subplots_layout(sample_df):
    """Test pour vérifier la disposition des sous-graphiques (subplots)"""
    with patch.object(plt, 'show'):
        Viz.plot_top_values(sample_df, nb_cols=2)

    fig = plt.gcf()
    axes = fig.get_axes()
    assert len(axes) > 1  # Vérifie qu'il y a plus d'un graphique (sous-graphiques)

def test_plot_top_values_empty_dataframe():
    """Test avec un DataFrame vide pour vérifier la gestion des cas vides."""
    empty_df = pd.DataFrame(columns=[('col1', 'value'), ('col1', 'count')])
    with pytest.raises(ValueError, match="Le dataset doit être alimenté."):
        Viz.plot_top_values(empty_df, nb_cols=1)

def test_plot_outliers_iqr_valid_input():
    """Test avec des entrées valides pour vérifier qu'aucune exception n'est levée."""
    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
        "feature2": [15, 14, 500, 15, 13],
    })
    outliers = pd.DataFrame({
        "feature1": [False, False, False, True, False],
        "feature2": [False, False, True, False, False],
    }, index=df.index)

    with patch("matplotlib.pyplot.show") as mock_show:
        Viz.plot_outliers_iqr(df, outliers, columns=["feature1", "feature2"])
        mock_show.assert_called_once()


def test_plot_outliers_iqr_empty_dataframe():
    """Test avec un DataFrame vide pour vérifier qu'une exception est levée."""
    df = pd.DataFrame()
    outliers = pd.DataFrame()

    with pytest.raises(ValueError, match="Le dataset doit être alimenté."):
        Viz.plot_outliers_iqr(df, outliers, columns=[])


def test_plot_outliers_iqr_single_column():
    """Test avec une seule colonne à visualiser."""
    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
    })
    outliers = pd.DataFrame({
        "feature1": [False, False, False, True, False],
    }, index=df.index)

    with patch("matplotlib.pyplot.show") as mock_show:
        Viz.plot_outliers_iqr(df, outliers, columns=["feature1"])
        mock_show.assert_called_once()


def test_plot_outliers_iqr_multiple_columns():
    """Test avec plusieurs colonnes pour vérifier la disposition des sous-graphiques."""
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
        Viz.plot_outliers_iqr(df, outliers, columns=["feature1", "feature2", "feature3"], nb_cols=2)
        mock_show.assert_called_once()

def test_plot_outliers_iqr_show_y_ticks():
    """Test avec l'option `show_y=True` pour vérifier l'ajout des ticks sur l'axe y."""
    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
    })
    outliers = pd.DataFrame({
        "feature1": [False, False, False, True, False],
    }, index=df.index)

    with patch("matplotlib.pyplot.show") as mock_show:
        Viz.plot_outliers_iqr(df, outliers, columns=["feature1"], show_y=True)
        mock_show.assert_called_once()
