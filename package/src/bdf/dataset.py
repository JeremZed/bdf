from bdf.tools import Tools

import pandas as pd
import numpy as np
import os

class Dataset:
    """
        Gestion du dataset
    """

    def __init__(self, data, options=None, **kwargs):
        """
        Initialise le dataset et charge les données.

        Args:
            data (str | pd.DataFrame | np.ndarray | list | dict):
                Données à charger. Peut être un chemin de fichier ou des données en mémoire.
            options (dict, optional):
                Options supplémentaires :
                - 'verbose' (int) : Niveau de détail pour les logs (par défaut 0).
                - 'name' (str) : Nom du dataset (par défaut, généré aléatoirement).
            kwargs : Arguments supplémentaires pour les fonctions de chargement.
        """
        self.options = options or {}

        self.verbose = self.options.get('verbose', 0)
        self.name = self.options.get('name', Tools.random_id())

        # raz des attributs et chargement des données
        self.reset().load_data(data, **kwargs)

    def reset(self):
        """
            Réinitialise les attributs du dataset.
        """
        Tools.log("Réinitialisation du dataset...", self.verbose)

        self.df = None

        Tools.log("Réinitialisation terminée.", self.verbose)

        return self

    def _load_from_file(self, filepath, **kwargs):
        """
        Charge des données à partir d'un fichier.

        Args:
            filepath (str): Chemin du fichier.
            kwargs : Arguments supplémentaires pour les fonctions de pandas.

        Returns:
            pd.DataFrame: DataFrame chargé à partir du fichier.
        """
        extension = filepath.split('.')[-1]
        loaders = {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "xlsx": pd.read_excel,
        }
        if extension not in loaders:
            raise ValueError(f"Extension non prise en charge : {extension}")

        try:
            return loaders[extension](filepath, **kwargs)
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier {filepath}: {e}")


    def load_data(self, data, **kwargs):
        """
        Charge les données et les transforme en DataFrame.

        Args:
            data (str | pd.DataFrame | np.ndarray | list | dict):
                Données à charger.
            kwargs : Arguments supplémentaires pour les fonctions de chargement.

        Raises:
            ValueError: Si le type de données n'est pas pris en charge.
            FileNotFoundError: Si le chemin du fichier est invalide.
        """

        Tools.log("Chargement des données...", self.verbose)

        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Fichier introuvable : {data}")
            self.df = self._load_from_file(data, **kwargs)

        elif isinstance(data, pd.DataFrame):
            self.df = data

        elif isinstance(data, np.ndarray):
            self.df = pd.DataFrame(data)

        elif isinstance(data, (list, dict)):
            self.df = pd.DataFrame(data)

        else:
            raise ValueError(f"Type de données non pris en charge : {type(data)}")

        Tools.log("Données chargées avec succès.", self.verbose)
        return self