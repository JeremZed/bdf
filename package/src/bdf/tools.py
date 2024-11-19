import os
from datetime import datetime
import random

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
                FileNotFoundError: Si le dossier spécifié dans `pathfile` n'existe pas.
                Exception: Si le chemin spécifié pour le fichier de log est invalide ou incorrect.

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
                        raise Exception(f"Chemin introuvable : {pathfile}")

    def random_id(length=12, pattern=None, uids=[], chars_special=None, excludes_chars=[]):
        """ Permet de générer un id aléatoirement """

        sequence_chars_upper = "ABCDEFGHIJKLMNOPQRSTUVWXTZ"
        sequence_chars_lower = sequence_chars_upper.lower()
        sequence_digits = "0123456789"
        sequence_chars_special = chars_special if chars_special is not None else "-_!?@$*."

        sequence_chars = sequence_chars_upper + sequence_chars_lower + sequence_digits + sequence_chars_special
        id = ""

        # Génération d'un id aléatoire standard respectant la taille passé en paramètre
        if pattern is None:
            for i in range(length):
                ch = Tools.get_random_element(sequence_chars, excludes_chars)
                id = id + ch
        else:
            for c in pattern:
                # Si il s'agit d'un caractère alphabétique en minuscule
                if c == "%s":
                    ch = Tools.get_random_element(sequence_chars_lower, excludes_chars)
                # Si il s'agit d'un caractère alphabétique en majuscule
                elif c == "%S":
                    ch = Tools.get_random_element(sequence_chars_upper, excludes_chars)
                # Si il s'agit d'un caractère numérique
                elif c == "%d":
                    ch = Tools.get_random_element(sequence_digits, excludes_chars)
                # Si il s'agit d'un caractère spécial
                elif c == "%x":
                    ch = Tools.get_random_element(sequence_chars_special, excludes_chars)
                else:
                    ch = c

                id = id + ch

        # On vérifie si l'id généré n'est pas un id déjà présent dans la liste
        if id in uids:
            return Tools.random_id(length=length, pattern=pattern, uids=uids)

        return id

    def get_random_element(items, excludes=[]):
        """
        Permet de retourner un élément aléatoire de la liste items tout faisant attention que l'élément sélectionné
        n'appartient pas à la liste des items à ne pas inclure
        """
        item = random.choice(items)

        return item if item not in excludes else Tools.get_random_element(items, excludes)
