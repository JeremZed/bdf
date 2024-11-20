import os
import pytest
from datetime import datetime
from unittest.mock import patch
from io import StringIO

from bdf.tools import Tools

LOG_FILENAME = "bdf.log"
LOG_FILE_PATH = "test_logs"  # Répertoire temporaire pour les tests de fichiers

@pytest.fixture(scope="module")
def create_log_directory():
    """Créer un répertoire pour stocker les logs pendant les tests"""
    if not os.path.exists(LOG_FILE_PATH):
        os.makedirs(LOG_FILE_PATH)
    yield
    # Nettoyage après les tests
    if os.path.exists(LOG_FILE_PATH):
        for file in os.listdir(LOG_FILE_PATH):
            os.remove(os.path.join(LOG_FILE_PATH, file))
        os.rmdir(LOG_FILE_PATH)


def test_log_to_console(capsys):
    """Test si le message de log est affiché correctement dans la console"""
    message = "Test log message"
    level = 1
    Tools.log(message, level, show=True, write=False, threshold=1)

    # Capturer la sortie de la console
    captured = capsys.readouterr()

    # Vérifier que le message est bien dans la sortie
    assert message in captured.out
    assert "[LOG]" in captured.out


def test_log_to_file(create_log_directory):
    """Test si le message de log est écrit correctement dans un fichier"""
    message = "Test log file"
    level = 1
    pathfile = os.path.join(LOG_FILE_PATH, LOG_FILENAME)

    Tools.log(message, level, show=False, write=True, threshold=1, pathfile=pathfile)

    # Vérifier si le fichier existe et contient le message
    assert os.path.exists(pathfile)
    with open(pathfile, 'r') as f:
        content = f.read()
    assert message in content


def test_log_with_invalid_directory():
    """Test si une exception est levée lorsque le dossier est introuvable"""
    message = "Test invalid directory"
    level = 1
    invalid_path = "invalid_directory"  # Dossier inexistant

    # Vérifier que l'exception est levée
    with pytest.raises(FileNotFoundError):
        Tools.log(message, level, show=False, write=True, threshold=1, pathfile=invalid_path)


def test_log_with_invalid_pathfile():
    """Test si une exception est levée pour un chemin de fichier incorrect"""
    message = "Test invalid pathfile"
    level = 1
    invalid_pathfile = "invalid_path_file.txt"  # Fichier invalide

    with pytest.raises(Exception):
        Tools.log(message, level, show=False, write=True, threshold=1, pathfile=invalid_pathfile)


def test_log_with_different_levels(capsys):
    """Test si le niveau de log fonctionne correctement avec le seuil"""
    message = "Test log with levels"
    level_info = 1
    level_warning = 2
    threshold = 2

    # Test avec un niveau inférieur au seuil (ne doit pas s'afficher)
    Tools.log(message, level_info, show=True, write=False, threshold=threshold)
    captured = capsys.readouterr()
    assert message not in captured.out

    # Test avec un niveau supérieur ou égal au seuil (doit s'afficher)
    Tools.log(message, level_warning, show=True, write=False, threshold=threshold)
    captured = capsys.readouterr()
    assert message in captured.out


@pytest.mark.parametrize("show, expected_output", [
    (True, "Test log message"),
    (False, "")
])
def test_log_show_parameter(capsys, show, expected_output):
    """Test le paramètre 'show' pour afficher ou non le log dans la console"""
    message = "Test log message"
    level = 1
    Tools.log(message, level, show=show, write=False, threshold=1)
    captured = capsys.readouterr()

    assert expected_output in captured.out

def test_default_generation():
    """Test génération par défaut avec les paramètres par défaut."""
    result = Tools.random_id()
    assert len(result) == 12
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz0123456789-_!?@$*."
    assert all(char in valid_chars for char in result)

def test_custom_length():
    """Test avec une longueur personnalisée."""
    length = 20
    result = Tools.random_id(length=length)
    assert len(result) == length

def test_custom_special_characters():
    """Test avec un ensemble personnalisé de caractères spéciaux."""
    chars_special = "!@#"
    result = Tools.random_id(chars_special=chars_special)
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz0123456789" + chars_special
    assert all(char in valid_chars for char in result)

def test_excludes_characters():
    """Test avec des caractères exclus."""
    excludes_chars = ['A', '1', '!', 'z']
    result = Tools.random_id(excludes_chars=excludes_chars)
    assert not any(char in excludes_chars for char in result)

def test_pattern_generation():
    """Test avec un pattern spécifique."""
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
    """Test pour éviter les doublons dans la liste des identifiants."""
    existing_ids = ["ABC123", "XYZ789"]
    new_id = Tools.random_id(length=6, uids=existing_ids)
    assert new_id not in existing_ids

def test_invalid_length():
    """Test avec une longueur invalide."""
    with pytest.raises(ValueError, match="le paramètre length doit être un entier positif."):
        Tools.random_id(length=-5)

def test_invalid_special_characters():
    """Test avec des caractères spéciaux invalides."""
    with pytest.raises(ValueError, match="le paramètre chars_special doit être une chaîne de caractère."):
        Tools.random_id(chars_special=123)

def test_invalid_pattern_type():
    """Test avec un pattern invalide (non-liste)."""
    with pytest.raises(ValueError, match="Le paramètre pattern doit une liste contenant au moins un élément."):
        Tools.random_id(pattern="invalid_pattern")

def test_empty_pattern():
    """Test avec un pattern vide."""
    with pytest.raises(ValueError, match="Le paramètre pattern doit une liste contenant au moins un élément."):
        Tools.random_id(pattern=[])

def test_all_characters_excluded():
    """Test où tous les caractères sont exclus."""
    excludes_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz0123456789-_!?@$*.")
    with pytest.raises(ValueError, match="Tous les caractères possibles sont exclus. Impossible de générer un identifiant."):
        Tools.random_id(excludes_chars=excludes_chars)

def test_no_valid_items_in_get_random_element():
    """Test avec une liste vide ou des exclusions impossibles pour get_random_element."""
    with pytest.raises(ValueError, match="Aucun élément n'est disponible pour la sélection."):
        Tools.get_random_element(["A", "B"], excludes=["A", "B"])

def test_recursive_id_generation():
    """Test de génération récursive avec des doublons initiaux."""
    existing_ids = ["ABCDEF" for _ in range(100)]  # Crée une situation potentielle de conflit
    new_id = Tools.random_id(length=6, uids=existing_ids)
    assert new_id not in existing_ids
