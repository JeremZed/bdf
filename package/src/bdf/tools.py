import os
from datetime import datetime

log_filename = "bdf.log"

class Tools:
    """
    Boîte à outils permettant d'effectuer différents traitement
    """

    @staticmethod
    def log(message, level, show=True, write=True, threshold=1, pathfile=f"{os.getcwd()}{os.sep}{log_filename}", raising=True):
        """
            Permet d'afficher un log en fonction d'un niveau de verbosité et d'un seuil
            Permet d'écrire le contenu du log dans un message
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
