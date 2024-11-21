import os
from datetime import datetime
import random

import numpy as np

log_filename = "bdf.log"

class Tools:
    """
    Classe utilitaire pour effectuer diverses opérations.

    Cette classe permet de gérer l'affichage et l'enregistrement des logs dans des fichiers.
    Elle offre une méthode pour loguer des messages avec différents niveaux de verbosité,
    et la possibilité de définir des seuils pour afficher ou écrire les logs dans un fichier.

    Attributs:
        log_filename (str): Nom par défaut du fichier de log (bdf.log).
    """

    @staticmethod
    def log(message, level, show=True, write=True, threshold=1, pathfile=f"{os.getcwd()}{os.sep}{log_filename}", raising=True):
        """
            Affiche et/ou écrit un message de log en fonction de son niveau et d'un seuil défini.

            Cette méthode permet de loguer un message en fonction de son niveau de verbosité. Elle peut
            afficher le log dans la console, l'écrire dans un fichier, ou les deux, selon les paramètres.

            Args:
                message (str): Le message à loguer.
                level (int): Le niveau de verbosité du log (par exemple, 1 pour info, 2 pour avertissement, etc.).
                show (bool, optionnel): Si True, affiche le message dans la console. Par défaut, True.
                write (bool, optionnel): Si True, écrit le message dans un fichier de log. Par défaut, True.
                threshold (int, optionnel): Le niveau minimum de verbosité pour que le log soit affiché/écrit. Par défaut, 1.
                pathfile (str, optionnel): Le chemin du fichier où écrire le log. Par défaut, "bdf.log" dans le répertoire actuel.
                raising (bool, optionnel): Si True, lève une exception lorsque le chemin est incorrect ou inaccessible. Par défaut, True.

            Raises:
                FileNotFoundError: Si le dossier/fichier n'existe pas.

            Exemple:
                >>> Tools.log("Un message de log", level=1, show=True, write=True, threshold=1)
                "Affiche le message dans la console et l'écrit dans le fichier bdf.log."
            """
        # Si le niveau de verbosité du log est supérieur ou égal à la condition de seuil
        if level >= threshold:
            dt = datetime.now()
            time_of_log = dt.strftime("%y-%m-%d %H:%M:%S:%f")[:-3]
            prefix = f"[LOG] {time_of_log}|{level}| "
            content_to_log = f"{prefix}{message}\n"

            # On affiche le log formaté
            if show :
                print(content_to_log)

            # Ecriture du message dans un fichier de log
            if write :
                # Si on indique uniquement un path de dossier
                # alors on créer le fichier bdf.log par défaut dans ce répertoire
                if os.path.isdir( pathfile )  :
                    os.path.join(pathfile, log_filename)

                if pathfile[-4:] == ".log":
                    dirname = os.path.dirname(pathfile)
                    # Check si le dossier passé en paramètre existe
                    if os.path.exists(dirname) :
                        with open(pathfile, "+a") as f:
                            f.write(content_to_log)
                    else:
                        if raising :
                            raise FileNotFoundError(f"Dossier introuvable : {dirname}")
                else:
                    if raising :
                        raise FileNotFoundError(f"Chemin introuvable : {pathfile}")

    def random_id(length=12, pattern=None, uids=None, chars_special=None, excludes_chars=None):
        """
            Génère un identifiant aléatoire selon des critères spécifiés.

            Args:
                length (int, optional):
                    Longueur de l'identifiant généré. Par défaut, 12.
                pattern (list, optional):
                    Modèle à utiliser pour générer l'identifiant. Chaque caractère du modèle peut correspondre à :
                        - "%s" : lettre minuscule.
                        - "%S" : lettre majuscule.
                        - "%d" : chiffre.
                        - "%x" : caractère spécial.
                        Tout autre caractère est utilisé tel quel.
                    Si `None`, un identifiant standard est généré sans contrainte.
                uids (list, optional):
                    Liste des identifiants existants pour éviter les doublons. Par défaut, une liste vide.
                chars_special (string, optional):
                    Ensemble des caractères spéciaux utilisables. Par défaut, `"-_!?@$*."`.
                excludes_chars (list, optional):
                    Liste des caractères à exclure lors de la génération. Par défaut, une liste vide.

            Returns:
                str: Identifiant généré qui respecte les critères spécifiés.

            Raises:
                RecursionError: Si un identifiant unique ne peut pas être généré après plusieurs tentatives.
                ValueError : Si un paramètre n'est pas du bon type.

            Notes:
                - Si aucun `pattern` n'est fourni, l'identifiant est généré en utilisant un mélange de lettres
                majuscules, minuscules, chiffres et caractères spéciaux.
                - Les caractères exclus dans `excludes_chars` ne sont jamais inclus dans l'identifiant généré.
                - Si l'identifiant généré existe déjà dans `uids`, la fonction est rappelée de manière récursive
                pour tenter de générer un nouvel identifiant unique.
        """

        sequence_chars_upper = "ABCDEFGHIJKLMNOPQRSTUVWXTZ"
        sequence_chars_lower = sequence_chars_upper.lower()
        sequence_digits = "0123456789"
        sequence_chars_special = chars_special if chars_special is not None else "-_!?@$*."

        if not isinstance(sequence_chars_special, str):
            raise ValueError("le paramètre chars_special doit être une chaîne de caractère.")

        if chars_special is not None:
            if not all(char.isprintable() and not char.isspace() for char in chars_special):
                raise ValueError("le paramètre chars_special contient des caractères non valides (espaces ou non imprimables).")

        if uids is None:
            uids = []

        if excludes_chars is None:
            excludes_chars = []

        if not isinstance(length, int) or length <= 0:
            raise ValueError("le paramètre length doit être un entier positif.")

        if not isinstance(uids, list):
            raise ValueError("le paramètre uids doit être une liste.")

        if not isinstance(excludes_chars, list):
            raise ValueError("le paramètre excludes_chars doit être une liste.")


        sequence_chars = sequence_chars_upper + sequence_chars_lower + sequence_digits + sequence_chars_special
        id = ""

        diff_chars = set(sequence_chars) - set(excludes_chars)
        if not diff_chars:
            raise ValueError("Tous les caractères possibles sont exclus. Impossible de générer un identifiant.")

        # Génération d'un id aléatoire standard respectant la taille passé en paramètre
        if pattern is None:
            for i in range(length):
                ch = Tools.get_random_element(sequence_chars, excludes_chars)
                id = id + ch
        else:

            if not isinstance(pattern, list) or len(pattern) == 0:
                raise ValueError("Le paramètre pattern doit une liste contenant au moins un élément.")

            all_sequences = {
                "%s": sequence_chars_lower,
                "%S": sequence_chars_upper,
                "%d": sequence_digits,
                "%x": sequence_chars_special
            }

            for c in pattern:
                if c in all_sequences:
                        ch = Tools.get_random_element(all_sequences[c], excludes_chars)
                else:
                    ch = c

                id = id + ch

        uids_set = set(uids)
        # On vérifie si l'id généré n'est pas un id déjà présent dans la liste
        if id in uids_set:
            return Tools.random_id(length, pattern, uids, chars_special, excludes_chars)

        return id

    def get_random_element(items, excludes=[]):
        """
            Retourne un élément aléatoire de la liste items, en excluant les éléments non autorisés.
        """
        elements = [item for item in items if item not in excludes]

        if not elements:
            raise ValueError("Aucun élément n'est disponible pour la sélection.")

        return random.choice(elements)

    def is_all_numeric(df):
        """
            Vérifie si toutes les colonnes d'un DataFrame sont de type numérique.

            Cette fonction inspecte les types de données des colonnes d'un DataFrame pour déterminer
            si elles sont toutes de type numérique (par exemple, int, float, etc.).

            Args:
                df (pd.DataFrame): Le DataFrame à analyser.

            Returns:
                bool:
                    - `True` si toutes les colonnes du DataFrame sont de type numérique.
                    - `False` sinon.

            Exemple:
                >>> import pandas as pd
                >>> import numpy as np
                >>> df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.6, 6.7]})
                >>> is_all_numeric(df1)
                True
                >>> df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
                >>> is_all_numeric(df2)
                False
            """

        numeric_columns = df.select_dtypes(include=np.number).columns
        return len(numeric_columns) == len(df.columns)

