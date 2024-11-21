from bdf.dataset import Dataset
from bdf.tools import Tools
from bdf.visualization import Viz

import pytest
import pandas as pd
import numpy as np
import os

from unittest.mock import patch, MagicMock

@pytest.fixture
def sample_csv(tmp_path):
    """Crée un fichier CSV temporaire pour les tests."""
    data = "col1,col2,col3\n1,2,3\n4,5,6"
    filepath = tmp_path / "sample.csv"
    filepath.write_text(data)
    return filepath

@pytest.fixture
def sample_json(tmp_path):
    """Crée un fichier JSON temporaire pour les tests."""
    data = [{"col1": 1, "col2": 2}, {"col1": 4, "col2": 5}]
    filepath = tmp_path / "sample.json"
    filepath.write_text(pd.DataFrame(data).to_json(orient="records"))
    return filepath

@pytest.fixture
def sample_excel(tmp_path):
    """Crée un fichier Excel temporaire pour les tests."""
    data = pd.DataFrame({"col1": [1, 4], "col2": [2, 5]})
    filepath = tmp_path / "sample.xlsx"
    data.to_excel(filepath, index=False)
    return filepath

def test_load_from_dataframe():
    """Test le chargement depuis un DataFrame."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data)
    assert dataset.df.equals(data)

def test_load_from_numpy_array():
    """Test le chargement depuis un tableau numpy."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    dataset = Dataset(data)
    assert dataset.df.shape == (2, 3)
    assert (dataset.df.values == data).all()

def test_load_from_list():
    """Test le chargement depuis une liste."""
    data = [[1, 2, 3], [4, 5, 6]]
    dataset = Dataset(data)
    assert dataset.df.shape == (2, 3)
    assert dataset.df.iloc[0, 0] == 1

def test_load_from_dict():
    """Test le chargement depuis un dictionnaire."""
    data = {"col1": [1, 4], "col2": [2, 5]}
    dataset = Dataset(data)
    assert dataset.df.shape == (2, 2)
    assert list(dataset.df.columns) == ["col1", "col2"]

def test_load_from_csv(sample_csv):
    """Test le chargement depuis un fichier CSV."""
    dataset = Dataset(str(sample_csv))
    assert dataset.df.shape == (2, 3)
    assert list(dataset.df.columns) == ["col1", "col2", "col3"]

def test_load_from_json(sample_json):
    """Test le chargement depuis un fichier JSON."""
    dataset = Dataset(str(sample_json))
    assert dataset.df.shape == (2, 2)
    assert list(dataset.df.columns) == ["col1", "col2"]

def test_load_from_excel(sample_excel):
    """Test le chargement depuis un fichier Excel."""
    dataset = Dataset(str(sample_excel))
    assert dataset.df.shape == (2, 2)
    assert list(dataset.df.columns) == ["col1", "col2"]

def test_file_not_found():
    """Test le cas où le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
        Dataset("invalid_path.csv")

def test_unsupported_extension(tmp_path):
    """Test le chargement d'un fichier avec une extension non supportée."""
    unsupported_file = tmp_path / "data.txt"
    unsupported_file.write_text("data")
    with pytest.raises(ValueError, match="Extension non prise en charge"):
        Dataset(str(unsupported_file))

def test_reset_method():
    """Test la méthode reset pour réinitialiser l'état."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data)
    dataset.reset()
    assert dataset.df is None

def test_invalid_data_type():
    """Test un type de données non supporté."""
    with pytest.raises(ValueError, match="Le type des données est invalide :"):
        Dataset(12345)

def test_logging_verbose(capfd):
    """Test que les logs s'affichent lorsque verbose est activé."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data, options={"verbose": 1})
    captured = capfd.readouterr()
    assert "Réinitialisation du dataset" in captured.out
    assert "Données chargées avec succès" in captured.out

def test_logging_not_verbose(capfd):
    """Test que les logs ne s'affichent pas lorsque verbose est désactivé."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = Dataset(data, options={"verbose": 0})
    captured = capfd.readouterr()
    assert captured.out == ""


@pytest.fixture
def sample_dataset():
    """Fixture pour créer un jeu de données d'exemple."""
    data = {
        "col1": [1, 2, 3, 4, 5],
        "col2": [5, 4, 3, 2, 1],
        "col3": [np.nan, 2, 3, np.nan, 5]
    }
    df = pd.DataFrame(data)
    return Dataset(df)

def test_head(sample_dataset):
    """Test de la fonction head()"""
    result = sample_dataset.head(3)
    assert result.shape == (3, 3)  # 3 premières lignes, 3 colonnes
    assert result["col1"].iloc[0] == 1  # Première ligne de col1 doit être 1

def test_tail(sample_dataset):
    """Test de la fonction tail()"""
    result = sample_dataset.tail(2)
    assert result.shape == (2, 3)  # 2 dernières lignes, 3 colonnes
    assert result["col1"].iloc[0] == 4  # Première ligne de col1 dans les 2 dernières lignes doit être 4

def test_info(sample_dataset):
    """Test de la fonction info()"""
    result = sample_dataset.info()  # info() ne retourne rien, vérifie la sortie dans la console manuellement
    assert result is None  # Il n'y a pas de valeur retournée par info()

def test_describe(sample_dataset):
    """Test de la fonction describe()"""
    result = sample_dataset.describe()
    assert "col1" in result.columns
    assert result["col1"]["mean"] == 3  # La moyenne de la colonne col1 doit être 3

def test_shape(sample_dataset):
    """Test de la fonction shape()"""
    result = sample_dataset.shape()
    assert result == (5, 3)  # Le dataset a 5 lignes et 3 colonnes

def test_missing_values(sample_dataset):
    """Test de la fonction missing_values()"""
    result = sample_dataset.missing_values()
    assert result.shape == (4, 2)  # 3 colonnes, 2 valeurs à afficher (ratio et sum)
    assert result["ratio"].iloc[0] == 2.0  # La proportion de valeurs manquantes dans col3

def test_drop_missing_values(sample_dataset):
    """Test de la fonction drop_missing_values()"""
    result = sample_dataset.drop_missing_values()
    assert result.df.shape == (3, 3)  # Après suppression des lignes avec NaN, il reste 3 lignes
    assert result.df.isna().sum().sum() == 0  # Il ne devrait plus y avoir de valeurs manquantes

def test_fill_missing(sample_dataset):
    """Test de la fonction fill_missing() avec stratégie 'mean'"""
    result = sample_dataset.fill_missing(strategy='mean')
    assert result.df["col3"].isna().sum() == 0  # Après remplissage, il ne devrait plus y avoir de NaN
    assert result.df["col3"].iloc[0] == 3.3333333333333335  # Valeur remplie par la moyenne (2+3+5)/3

def test_duplicated_values(sample_dataset):
    """Test de la fonction duplicated_values()"""
    result = sample_dataset.duplicated_values()
    assert result == 0  # Aucune valeur dupliquée dans le dataset

    # Création d'un doublon
    sample_dataset.df.loc[len(sample_dataset.df.index)] = [3,3,3]

    result = sample_dataset.duplicated_values()
    assert result == 2  # Maintenant, il y a un doublon


def test_drop_duplicated_values(sample_dataset):
    """Test de la fonction drop_duplicated_values()"""
    sample_dataset.df.loc[len(sample_dataset.df.index)] = [3,3,3]  # Création d'un doublon
    sample_dataset.drop_duplicated_values()

    assert sample_dataset.df.shape == (5, 3)  # Après suppression du doublon, il devrait y avoir 4 lignes

def test_dtypes(sample_dataset):
    """Test de la fonction dtypes()"""
    result = sample_dataset.dtypes()
    assert result["col1"] == np.int64
    assert result["col2"] == np.int64
    assert result["col3"] == float

    result_count = sample_dataset.dtypes(mode="count")

    assert result_count.iloc[0] == 2  # Il y a 2 colonnes de type int64

def test_convert_dtypes(sample_dataset):
    """Test de la fonction convert_dtypes()"""
    result = sample_dataset.convert_dtypes({"col1": np.float64})
    assert result.df["col1"].dtype == np.float64  # La colonne col1 doit être convertie en float64

def test_normalize(sample_dataset):
    """Test de la fonction normalize()"""
    result = sample_dataset.normalize(columns=["col1"])
    assert result.df["col1"].min() == 0
    assert result.df["col1"].max() == 1

def test_standardize(sample_dataset):
    """Test de la fonction standardize()"""
    result = sample_dataset.standardize(columns=["col1"])
    assert result.df["col1"].mean() == pytest.approx(0, 1e-6)  # La moyenne doit être proche de 0
    assert result.df["col1"].std() == pytest.approx(1, 1e-6)  # L'écart-type doit être proche de 1

def test_value_counts(sample_dataset):
    """Test de la fonction value_counts()"""
    result = sample_dataset.value_counts("col1")
    assert result[1] == 1  # La valeur 1 apparaît 1 fois
    assert result[5] == 1  # La valeur 5 apparaît 1 fois

def test_correlations(sample_dataset):
    """Test de la fonction correlations()"""
    result = sample_dataset.correlations()
    assert result.shape == (3, 3)  # Matrice de corrélation de 3x3
    assert result["col1"]["col2"] == -1  # La corrélation entre col1 et col2 doit être -1

def test_top_values(sample_dataset):
    """Test de la fonction top_values()"""
    result = sample_dataset.top_values(n=2)
    assert result.shape == (2, 6)  # Il y a 2 lignes et 6 colonnes (2 valeurs + 2 comptages par colonne)
    assert result["col1"]["value"].iloc[0] == 1  # La première valeur de col1 est 1

def test_filter_rows(sample_dataset):
    """Test de la fonction filter_rows()"""
    result = sample_dataset.filter_rows("col1 > 2")
    assert result.shape == (3, 3)  # 3 lignes doivent respecter la condition
    assert result["col1"].iloc[0] == 3  # La première valeur de col1 après filtrage doit être 3

def test_add_column(sample_dataset):
    """Test de la fonction add_column()"""
    result = sample_dataset.add_column("col4", [10, 20, 30, 40, 50])
    assert "col4" in result.df.columns  # La nouvelle colonne doit exister
    assert result.df["col4"].iloc[0] == 10  # La première valeur de col4 doit être 10


@pytest.fixture
def setup_method():
    """Fixture pour créer un jeu de données d'exemple."""

    df = pd.DataFrame({
        "feature1": [10, 12, 13, 500, 11],
        "feature2": [15, 14, 500, 15, 13],
        "feature3": [1.2, 2.5, 3.1, 4.8, 5.5],
        "non_numeric": ["A", "B", "C", "D", "E"]
    })
    return Dataset(df)


def test_outliers_iqr(setup_method):
    """Test de base avec la méthode IQR."""
    result = setup_method.outliers(columns=["feature1", "feature3"], method="IQR")
    assert isinstance(result, pd.DataFrame)
    assert "count" in result.columns
    assert "ratio" in result.columns
    assert result.loc["feature1", "count"] == 1
    assert result.loc["feature3", "count"] == 0

def test_outliers_with_columns(setup_method):
    """Test en spécifiant une liste de colonnes."""
    result = setup_method.outliers(columns=["feature1", "feature3"], method="IQR")
    assert "feature2" not in result.index
    assert "feature1" in result.index
    assert "feature3" in result.index

def test_outliers_empty_dataframe(setup_method):
    """Test avec un DataFrame vide."""
    empty_df = pd.DataFrame()
    instance = Dataset(empty_df)
    with pytest.raises(ValueError, match="Le dataset doit être alimenté."):
        instance.outliers()

def test_outliers_non_numeric_columns(setup_method):
    """Test avec des colonnes non numériques."""
    with pytest.raises(ValueError, match="Les colonnes de sont pas toutes de type numérique."):
        setup_method.outliers(columns=["non_numeric"])

def test_outliers_invalid_method(setup_method):
    """Test avec une méthode invalide."""
    with pytest.raises(ValueError, match="La méthode de calcul n'est pas prise en charge : invalid_method"):
        setup_method.outliers(method="invalid_method")

def test_outliers_show_graph(setup_method):
    """Test avec l'affichage des graphiques activé."""
    with patch("bdf.visualization.Viz.plot_outliers_iqr") as mock_plot:
        setup_method.outliers(show_graph=True)
        mock_plot.assert_called_once()

def test_outliers_ratio_calculation(setup_method):
    """Test du calcul du ratio des outliers."""
    result = setup_method.outliers(method="IQR")
    assert result.loc["feature1", "ratio"] == 20.0
    assert result.loc["feature2", "ratio"] == 20.0
    assert result.loc["feature3", "ratio"] == 0.0
    assert result.loc["BDF_total_of_values", "ratio"] == 10

def test_outliers_custom_threshold(setup_method):
    """Test avec un paramètre personnalisé pour le seuil (threshold)."""
    result = setup_method.outliers(method="IQR", threshold=3)
    assert result.loc["feature1", "count"] == 1
    assert result.loc["feature3", "count"] == 0  # Aucun outlier avec un seuil plus élevé.