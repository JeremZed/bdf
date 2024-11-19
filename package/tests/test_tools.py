import os
import pytest
from datetime import datetime
from unittest.mock import patch
from io import StringIO

# Assurez-vous que l'import de la classe Tools est correct
from bdf.tools import Tools  # Remplacer par le nom du fichier ou module de la classe

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

    # Appeler la méthode log pour écrire dans le fichier
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
