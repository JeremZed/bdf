from bdf.tools import Tools
from bdf.visualization import Viz
from bdf.outliers import Outlier

import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

class Dataset:
    """
    Classe représentant un dataset et fournissant des outils pour manipuler, nettoyer et analyser les données.

    Cette classe permet de charger des données à partir de différents formats (CSV, JSON, Excel, etc.), d'effectuer diverses opérations de nettoyage (gestion des valeurs manquantes, doublons), de transformation (normalisation, standardisation, conversion de types), ainsi que de calculs statistiques et de visualisation.

    Attributes:
        df (pd.DataFrame): Le DataFrame contenant les données du dataset.
        options (dict): Options de configuration pour la gestion du dataset, comme le niveau de verbosité et le nom du dataset.
        verbose (int): Niveau de détail pour les logs (par défaut 0).
        name (str): Nom du dataset.

    Methods:
        reset(): Réinitialise les attributs du dataset.
        load_data(data, **kwargs): Charge les données dans le dataset à partir d'une source (fichier, DataFrame, etc.).
        head(n=5): Retourne les premières n lignes du dataset.
        tail(n=5): Retourne les dernières n lignes du dataset.
        info(**kwargs): Retourne des informations détaillées sur le dataset.
        describe(**kwargs): Retourne un résumé statistique des colonnes du dataset.
        shape(): Retourne la forme du dataset (nombre de lignes et colonnes).
        missing_values(show_heatmap=False, figsize=(8,8)): Affiche la proportion de valeurs manquantes et éventuellement une heatmap.
        drop_missing_values(**kwargs): Supprime les lignes contenant des valeurs manquantes.
        fill_missing(strategy=0, columns=None): Remplie les valeurs manquantes par la moyenne, la médiane ou une valeur spécifique.
        duplicated_values(filter=None, show=False): Affiche ou retourne le nombre de valeurs dupliquées.
        drop_duplicated_values(**kwargs): Supprime les lignes contenant des doublons.
        dtypes(mode=None): Retourne les types de données des colonnes ou leur comptabilisation.
        convert_dtypes(dtype_dict): Convertit les types de données des colonnes selon un dictionnaire spécifié.
        normalize(columns=None): Applique une normalisation Min-Max sur les colonnes spécifiées.
        standardize(columns=None): Applique une standardisation Z-Score sur les colonnes spécifiées.
        value_counts(column): Retourne le nombre d'occurrences de chaque valeur unique dans une colonne donnée.
        correlations(show_heatmap=False, figsize=(8,8)): Affiche la matrice de corrélation des colonnes numériques et une heatmap optionnelle.
        top_values(n=10, filter=None): Retourne les n valeurs les plus fréquentes pour chaque colonne ou filtrées par colonne.
        plot_histogram(column, bins=10, figsize=(8,8)): Affiche un histogramme pour une colonne donnée.
        plot_boxplot(column, bins=10, figsize=(8,8)): Affiche un boxplot pour une colonne donnée.
        filter_rows(condition): Filtre les lignes du dataset en fonction d'une condition spécifique.
        add_column(column_name, values): Ajoute une nouvelle colonne au dataset.
        merge(other, on, how='inner'): Fusionne le dataset avec un autre Dataset ou DataFrame.
        sample(n=5): Retourne un échantillon aléatoire de n lignes du dataset.
        to_csv(filepath): Sauvegarde le dataset en fichier CSV.
        to_excel(filepath): Sauvegarde le dataset en fichier Excel.
        to_json(filepath): Sauvegarde le dataset en fichier JSON.
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

        if not isinstance(data, (str, pd.DataFrame, np.ndarray, list, dict)):
            raise ValueError(f"Le type des données est invalide : {type(data)}")

        self.options = options or {}

        self.verbose = self.options.get('verbose', 0)
        self.name = self.options.get('name', Tools.random_id())

        # raz des attributs et chargement des données
        self.reset().load_data(data, **kwargs)

    def reset(self):
        """
        Réinitialise les attributs du dataset.

        Returns:
            Dataset: L'instance actuelle du dataset après réinitialisation.
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

        Raises:
            ValueError: Si le format du fichier n'est pas pris en charge.
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

        Returns:
            Dataset: L'instance actuelle du dataset après chargement des données.
        """

        Tools.log("Chargement des données...", self.verbose)

        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Fichier introuvable : {data}")
            self.df = self._load_from_file(data, **kwargs)

        elif isinstance(data, pd.DataFrame):
            self.df = data

        elif isinstance(data, (list, np.ndarray)):
            self.df = pd.DataFrame(data)

        elif isinstance(data, dict):
            self.df = pd.DataFrame.from_dict(data)

        else:
            raise ValueError(f"Type de données non pris en charge : {type(data)}")

        Tools.log("Données chargées avec succès.", self.verbose)
        return self

    def head(self, n=5):
        """
        Permet d'afficher les n premières lignes du dataset.

        Args:
            n (int, optional): Nombre de lignes à afficher (par défaut 5).

        Returns:
            pd.DataFrame: Les premières n lignes du dataset.
        """
        return self.df.head(n)

    def tail(self, n=5):
        """
        Permet d'afficher les n dernières lignes du dataset.

        Args:
            n (int, optional): Nombre de lignes à afficher (par défaut 5).

        Returns:
            pd.DataFrame: Les dernières n lignes du dataset.
        """
        return self.df.tail(n)

    def info(self, **kwargs):
        """
        Permet de retourner les informations du dataset.

        Args:
            **kwargs : Arguments supplémentaires à passer à la fonction pandas `info`.

        Returns:
            None: Affiche les informations du dataset.
        """
        return self.df.info(**kwargs)

    def describe(self, **kwargs):
        """
        Permet de retourner un résumé statistique des colonnes, par défaut numérique.

        Args:
            **kwargs : Arguments supplémentaires à passer à la fonction pandas `describe`.

        Returns:
            pd.DataFrame: Résumé statistique du dataset.
        """
        return self.df.describe(**kwargs)

    def shape(self):
        """
        Permet de retourner la forme du dataset.

        Returns:
            tuple: La forme du dataset sous forme de tuple (nombre de lignes, nombre de colonnes).
        """
        return self.df.shape

    def missing_values(self, show_heatmap=False,figsize=(8,8)):
        """
        Permet de visualiser la proportion de valeurs manquantes dans le dataset.

        Args:
            show_heatmap (bool, optional): Si True, affiche une heatmap des valeurs manquantes.
            figsize (tuple, optional): Taille de la figure pour la heatmap (par défaut (8, 8)).

        Returns:
            pd.DataFrame: Proportion des valeurs manquantes pour chaque colonne.
        """
        a = self.df.isna().sum() / self.df.shape[0]
        df = pd.DataFrame(a, columns=['ratio'])
        df['count'] = self.df.isna().sum()

        df.loc['BDF_total_of_values'] = [ df['count'].sum(), round((df['count'].sum() * 100) / (self.df.shape[0] * self.df.shape[1]), 2)  ]

        if show_heatmap == True:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
            sns.heatmap(self.df.isna(), cbar=True , annot=False, cmap="coolwarm", fmt="0.2f")
            plt.title("Heatmap des valeurs manquantes")
            plt.tight_layout()
            plt.show()

        return df.sort_values('ratio', ascending=False)

    def drop_missing_values(self, **kwargs):
        """
        Supprime les lignes contenant des valeurs manquantes dans le dataset.

        Args:
            **kwargs : Arguments supplémentaires pour la fonction pandas `dropna`.

        Returns:
            Dataset: L'instance actuelle du dataset après suppression des lignes.
        """
        self.df = self.df.dropna(**kwargs)

        return self

    def fill_missing(self, strategy=0, columns=None):
        """
        Remplir les valeurs manquantes dans le dataset.

        Args:
            strategy (int, str, optional): La stratégie de remplissage ('mean' pour la moyenne, 'median' pour la médiane,
                                      un entier ou un chiffre à virgule pour une valeur spécifique).
            columns (list, optional): Liste des colonnes à remplir (par défaut, toutes les colonnes).

        Returns:
            Dataset: L'instance actuelle du dataset après remplissage des valeurs manquantes.
        """

        if columns is None:
            columns = self.df.columns

        if strategy == 'mean':
            self.df[columns] = self.df[columns].fillna(self.df[columns].mean())
        elif strategy == 'median':
            self.df[columns] = self.df[columns].fillna(self.df[columns].median())
        elif isinstance(strategy, (int, float)):
            self.df[columns] = self.df[columns].fillna(strategy)

        return self

    def duplicated_values(self, filter=None, show=False):
        """
        Permet de détecter et compter les valeurs dupliquées dans le dataset.

        Args:
            filter (str, optional): Nom de la colonne à filtrer pour les doublons (par défaut, recherche dans toutes les colonnes).
            show (bool, optional): Si True, affiche les occurences en doublons (par défaut, False).

        Returns:
            pd.DataFrame or int: Le nombre de doublons dans le dataset ou un DataFrame avec les doublons filtrés.
        """

        if filter is not None:
            if not isinstance(filter, list):
                raise ValueError(f"Le filtre doit être une liste de feature.")

            items = self.df.loc[ self.df[filter].duplicated(keep=False), : ]

        else:
            items = self.df.loc[self.df.duplicated(keep=False), :]

        if show:
            return items
        else:
            return len(items)

    def drop_duplicated_values(self, **kwargs):
        """
        Supprime les lignes contenant des valeurs dupliquées dans le dataset.

        Args:
            **kwargs : Arguments supplémentaires pour la fonction pandas `drop_duplicates`.

        Returns:
            Dataset: L'instance actuelle du dataset après suppression des doublons.
        """
        self.df = self.df.drop_duplicates(**kwargs)

    def dtypes(self, mode=None):
        """
        Retourne les types de données des colonnes du dataset.

        Args:
            mode (str, optional): Si 'count', retourne la comptabilisation des types de données (par défaut, retourne les types exacts des colonnes).

        Returns:
            pd.Series or pd.DataFrame: Les types des colonnes ou une comptabilisation des types.
        """
        if mode == "count":
            return self.df.dtypes.value_counts()
        else:
            return self.df.dtypes

    def convert_dtypes(self, dtype_dict):
        """
        Convertit les types de données des colonnes du dataset.

        Args:
            dtype_dict (dict): Un dictionnaire où les clés sont les noms des colonnes et les valeurs sont les types cibles.

        Returns:
            Dataset: L'instance actuelle du dataset après conversion des types.
        """
        self.df = self.df.astype(dtype_dict)
        return self

    def normalize(self, columns=None):
        """
        Normalise les colonnes spécifiées du dataset à l'échelle [0, 1] en utilisant la méthode Min-Max.

        Args:
            columns (list, optional): Liste des colonnes à normaliser (par défaut, normalise toutes les colonnes numériques).

        Returns:
            Dataset: L'instance actuelle du dataset après normalisation.
        """
        columns = self._get_columns_by_type(columns)

        self.df[columns] = self.df[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return self

    def standardize(self, columns=None):
        """
        Standardise les colonnes spécifiées du dataset à la moyenne 0 et à l'écart-type 1.

        Args:
            columns (list, optional): Liste des colonnes à standardiser (par défaut, standardise toutes les colonnes numériques).

        Returns:
            Dataset: L'instance actuelle du dataset après standardisation.
        """

        columns = self._get_columns_by_type(columns)

        self.df[columns] = self.df[columns].apply(lambda x: (x - x.mean()) / x.std())
        return self

    def value_counts(self, column):
        """
        Retourne le nombre d'occurrences de chaque valeur unique dans une colonne donnée.

        Args:
            column (str): Le nom de la colonne dont on veut obtenir les valeurs uniques.

        Returns:
            pd.Series: Nombre d'occurrences de chaque valeur unique dans la colonne.
        """
        return self.df[column].value_counts()

    def correlations(self, show_heatmap=False, figsize=(8,8)):
        """
        Calcule et affiche la matrice de corrélation entre les colonnes numériques du dataset.

        Args:
            show_heatmap (bool, optional): Si True, affiche une heatmap de la matrice de corrélation (par défaut, False).
            figsize (tuple, optional): Taille de la figure pour la heatmap (par défaut (8, 8)).

        Returns:
            pd.DataFrame: Matrice de corrélation des colonnes numériques.
        """
        c = self.df.corr()

        if show_heatmap:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
            sns.heatmap(c, cbar=True , annot=True, cmap="coolwarm", fmt="0.2f", ax=ax)
            plt.title("Heatmap des corrélations")
            plt.tight_layout()
            plt.show()

        return c

    def top_values(self, n=10, filter=None, show_graph=False, nb_cols=3, w_graph=5, h_graph=5,figsize=None, show_y=False):
        """
        Retourne les n valeurs les plus fréquentes pour chaque colonne ou une colonne spécifique.

        Args:
            n (int, optional): Nombre de valeurs à retourner (par défaut, 10).
            filter (str, optional): Nom de la colonne à filtrer pour les top valeurs (par défaut, prend toutes les colonnes).

        Returns:
            pd.Series or pd.DataFrame: Les top n valeurs pour chaque colonne ou pour la colonne filtrée.
        """

        if filter is None:
            filter = self.df.columns

        data = {}

        for column in filter:

            counts = self.df[column].value_counts().head(n)

            data[column] = {
                "value": counts.index.tolist(),
                "count": counts.values.tolist(),
            }

        # le double en-tête
        header = pd.MultiIndex.from_product([filter, ['value', 'count']])

        rows = []
        for i in range(n):
            row = []
            for column in filter:
                values = data[column]["value"]
                counts = data[column]["count"]

                # si on dépasse la taille des value_counts
                row.extend([values[i] if i < len(values) else None,
                            counts[i] if i < len(counts) else None])
            rows.append(row)

        result = pd.DataFrame(rows, columns=header)

        if show_graph:
            Viz.plot_top_values(result, nb_cols=nb_cols, w_graph=w_graph, h_graph=h_graph, figsize=figsize, show_y=show_y)

        return result

    def filter_rows(self, condition):
        """
        Filtre les lignes du dataset selon une condition spécifique.

        Args:
            condition (str or callable): La condition à appliquer pour filtrer les lignes.

        Returns:
            pd.DataFrame: Un DataFrame contenant uniquement les lignes qui respectent la condition.
        """
        try:
            return self.df.query(condition)
        except Exception as e:
            raise ValueError(f"Condition invalide : {condition}. Erreur : {str(e)}")

    def add_column(self, column_name, values):
        """
        Ajoute une nouvelle colonne au dataset.

        Args:
            column_name (str): Le nom de la nouvelle colonne.
            values (list or pd.Series): Les valeurs de la colonne à ajouter.

        Returns:
            Dataset: L'instance actuelle du dataset après ajout de la nouvelle colonne.
        """
        self.df[column_name] = values
        return self

    def merge(self, other, on, how='inner'):
        """
        TODO Revoir les arguments de la fonction pour être flexible avec la fonction de pandas et la tester

        Fusionne le dataset avec un autre Dataset ou DataFrame.

        Args:
            other (Dataset or pd.DataFrame): L'autre dataset ou DataFrame à fusionner.
            on (str or list): Nom de la ou des colonnes sur lesquelles fusionner.
            how (str, optional): Méthode de fusion (par défaut, 'inner').

        Returns:
            Dataset: L'instance actuelle du dataset après fusion.
        """
        if isinstance(other, Dataset):
            self.df = pd.merge(self.df, other.df, on=on, how=how)
        else:
            self.df = pd.merge(self.df, other, on=on, how=how)

        return self

    def sample(self, n=5):
        """
        Retourne un échantillon aléatoire de n lignes du dataset.

        Args:
            n (int, optional): Nombre de lignes à échantillonner (par défaut, 5).

        Returns:
            pd.DataFrame: Un échantillon aléatoire de n lignes du dataset.
        """
        return self.df.sample(n)

    def to_csv(self, filepath):
        """
        Sauvegarde le dataset en fichier CSV.

        Args:
            filepath (str): Le chemin du fichier où sauvegarder le dataset.

        Returns:
            None: Sauvegarde le dataset dans un fichier CSV.
        """
        self.df.to_csv(filepath, index=False)

    def to_excel(self, filepath):
        """
        Sauvegarde le dataset en fichier Excel.

        Args:
            filepath (str): Le chemin du fichier où sauvegarder le dataset.

        Returns:
            None: Sauvegarde le dataset dans un fichier Excel.
        """
        self.df.to_excel(filepath, index=False)

    def to_json(self, filepath):
        """
        Sauvegarde le dataset en fichier JSON.

        Args:
            filepath (str): Le chemin du fichier où sauvegarder le dataset.

        Returns:
            None: Sauvegarde le dataset dans un fichier JSON.
        """
        self.df.to_json(filepath, orient="records")

    def _get_columns_by_type(self, columns=None, type_cols=[np.number]):
        """
        Retourne les colonnes du dataset à traiter, en fonction des paramètres spécifiés.

        Si le paramètre `columns` est `None`, la fonction sélectionne toutes les colonnes de types spécifiés dans `type_cols`.

        Args:
            columns (list, optional): Liste des colonnes spécifiques à traiter. Si `None`, toutes les colonnes de type `type_cols` sont sélectionnées.
            type_cols (list, optional): Liste des types de colonnes à sélectionner. Par défaut, sélectionne les colonnes numériques (type `np.number`).

        Returns:
            list: Liste des noms des colonnes à traiter, soit celles spécifiées dans `columns`, soit toutes les colonnes correspondant à `type_cols`.
        """
        if columns is None:
            return self.df.select_dtypes(include=type_cols).columns.tolist()

        return columns

    def has_only_features_numeric(self):
        """
            Permet de savoir si le dataset comporte uniquement des features numérique
        """

        return Tools.is_all_numeric(self.df)

    def outliers(self, columns=None, method="IQR", show_graph=False, nb_cols=2, w_graph=5, h_graph=5, show_y=False, figsize=None, **kwargs):
        """
            Identifie et retourne les outliers d'un DataFrame selon la méthode spécifiée.

            Cette fonction calcule les valeurs aberrantes (outliers) pour les colonnes numériques d'un DataFrame,
            ou pour un sous-ensemble de colonnes spécifié. Par défaut, la méthode utilisée est l'IQR (Interquartile Range).
            La fonction peut également afficher les graphiques des outliers sous forme de boxplots.

            Args:
                columns (list[str], optional): Liste des colonnes à analyser. Si `None`, toutes les colonnes numériques
                    du DataFrame sont utilisées. Par défaut : `None`.
                method (str, optional): Méthode pour calculer les outliers. Actuellement, seule la méthode `"IQR"` est prise en charge.
                    Par défaut : `"IQR"`.
                show_graph (bool, optional): Si `True`, affiche les graphiques des outliers sous forme de boxplots.
                    Par défaut : `False`.
                nb_cols (int, optional): Nombre de boxplots affichés par ligne dans le graphique. Par défaut : 2.
                w_graph (int, optional): Largeur de chaque boxplot en pouces. Par défaut : 5.
                h_graph (int, optional): Hauteur de chaque boxplot en pouces. Par défaut : 5.
                show_y (bool, optional): Si `True`, affiche les graduations de l'axe y sur les boxplots. Par défaut : `False`.
                figsize (tuple, optional): Taille de la figure `(largeur, hauteur)` en pouces. Si spécifié, remplace
                    `w_graph` et `h_graph`. Par défaut : `None`.
                **kwargs: Arguments supplémentaires passés à la méthode choisie (par exemple, `threshold` pour l'IQR).

            Raises:
                ValueError: Si la méthode spécifiée n'est pas prise en charge.
                ValueError: Si le DataFrame contient des colonnes non numériques.
                ValueError: Si les outliers ne peuvent pas être calculés pour une raison quelconque.

            Returns:
                pd.DataFrame: Un DataFrame contenant :
                    - `count`: Le nombre d'outliers pour chaque colonne.
                    - `ratio`: Le pourcentage d'outliers par rapport au total des lignes du DataFrame.
                    - Une ligne supplémentaire "BDF_total_of_values" avec le total des outliers et le ratio global.

            Affichage:
                Si `show_graph` est activé, des boxplots avec les valeurs aberrantes mises en évidence en rouge
                sont affichés.

            Exemple:
                >>> df = pd.DataFrame({
                ...     "feature1": [10, 12, 13, 500, 11],
                ...     "feature2": [15, 14, 500, 15, 13],
                ... })
                >>> obj = YourClass(df)
                >>> outliers = obj.outliers(columns=["feature1", "feature2"], method="IQR", show_graph=True)
                >>> print(outliers)
                            count  ratio
                feature1       1  20.0
                feature2       1  20.0
                BDF_total_of_values  2  20.0

            Notes:
                - D'autres méthodes comme Z-score, Isolation Forest, ou One-Class SVM peuvent être ajoutées dans le futur.
                - La ligne "BDF_total_of_values" représente les totaux combinés des outliers et leur ratio global.
            """

        if method not in ['IQR']:
            raise ValueError(f"La méthode de calcul n'est pas prise en charge : {method}")

        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()

        if Tools.is_all_numeric(self.df[columns]) == False:
            raise ValueError(f"Les colonnes de sont pas toutes de type numérique.")

        outliers = None
        if method == "IQR":
            outliers = Outlier.iqr(self.df, columns, **kwargs)

        # TODO method : Z-score
        # TODO method : Méthode de Tukey
        # TODO method : Isolation Forest
        # TODO method : One-Class SVM
        # TODO method : Méthode des Quartiles et Découpage par Décile
        # TODO method : DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        # TODO method : Clustering K-means
        # TODO method : Méthodes basées sur la régression (par exemple, régression robuste)

        if outliers is None:
            raise ValueError("Impossible de calculer les outliers.")

        if show_graph:
            Viz.plot_outliers_iqr(self.df, outliers, columns, nb_cols=nb_cols, w_graph=w_graph, h_graph=h_graph, figsize=figsize, show_y=show_y)

        result = pd.DataFrame(outliers.sum(), columns=['count'])
        result['ratio'] = result['count'].apply(lambda x : round((x * 100) / len(self.df), 2) )

        result.loc['BDF_total_of_values'] = [ result['count'].sum(), round((result['count'].sum() * 100) / (self.df.shape[0] * self.df.shape[1]), 2)  ]

        return result