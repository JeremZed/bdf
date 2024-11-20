from bdf.dataset import Dataset
from bdf.tools import Tools

import pytest
import pandas as pd
import numpy as np
import os

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
    with pytest.raises(ValueError, match="Type de données non pris en charge"):
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
