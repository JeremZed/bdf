from urllib.parse import urlparse
from dateutil import parser
import os

class FeatureTypeNumerical:

    tag = "numerical"

    instances = [
        np.int8, np.int16, np.int32, np.int64,
        np.float16, np.float32, np.float64,
        np.uint, np.uint16, np.uint32, np.uint64, np.uint8,
        int, float
    ]

    @staticmethod
    def check(series, mode=None):
        """ Permet de lancer le contrôle du type numérique sur toute la serie passée en paramètre """
        if series.dtype in FeatureTypeNumerical.instances:
            return True

        if mode == FeatureType.mode_convert:
            # Dans le cas où la Series n'est pas considérée comme une série constitué uniquement de valeur numérique
            # On tente un cast dans l'hypothèse où les valeurs ne soient pas dans le bon "type" mais représentent un type numéric tout de même
            # On passe par une chaine de caractères pour eviter de considérer les booleens comme un numérique
            try:
                _ = [ float(f"{x}") for x in series ]
            except:
                #Si le cas n'est pas possible alors la feature n'est pas considéré comme numérique
                return False
            return True

        return False

class FeatureTypeBoolean:
    tag = "boolean"

    instances = [np.bool, bool]

    @staticmethod
    def check(series, mode=None):
        """ Permet de lancer le contrôle du type boolean sur toute la serie passée en paramètre """
        if series.dtype in FeatureTypeBoolean.instances:
            return True

        if mode == FeatureType.mode_convert:
            # Dans le cas où la Series n'est pas considérée comme une série de type Boolean
            # On tente de vérifie si son contenu n'a pas besoin d'une conversion
            # Par exemple 0, "0", "False", false, False -> False
            # Par exemple 1, "1", "True", true, True -> True
            for x in series:
                if x not in [0, "0", "False", False, "false", 1, "1", "True", True, "true"]:
                    return False
            return True

        return False

class FeatureTypeURL:
    tag = "url"

    instances = [np.str_, str, np.object_]

    @staticmethod
    def check(series, mode=None):
        """ Permet de lancer le contrôle du type URL sur toute la serie passée en paramètre """
        if series.dtype in FeatureTypeURL.instances:
            for x in series:
                u = urlparse(f"{x}")
                if (u.netloc == "" or u.scheme == "") and u.path != "":
                    return False
            return True

        return False

class FeatureTypePath:
    tag = "path"

    instances = [np.str_, str, np.object_]
    extension_not_allow = ['com', 'fr', 'net', 'de']

    @staticmethod
    def check(series, mode=None):
        """ Permet de lancer le contrôle du type URL sur toute la serie passée en paramètre """
        if series.dtype in FeatureTypePath.instances:
            for x in series:
                u = urlparse(f"{x}")
                if (u.netloc == "" and u.scheme == "") and u.path != "" and FeatureTypePath.is_file_path(u.path) == False:
                    return False
            return True

        return False
    @staticmethod
    def is_file_path(p):
        """ Permet de vérifier si la chaine de caractères passée en paramètre correspond à un path de fichier """
        # # SI aucun séparateur présent alors on considère qu'il ne s'agit pas d'un fichier
        # if not any([ sep in p for sep in ['/', '\\'] ]):
        #     return False
        explode = p.split(".")
        print(len(explode))
        # SI une extension est présente dans le nom de base du path alors il s'agit d'un fichier
        if len(explode) > 0:
            last = explode[-1]
            if last != "" and last not in FeatureTypePath.extension_not_allow:
                return True

        return False


class FeatureTypeDate:
    tag = "date"
    instances = [np.datetime64]

    @staticmethod
    def check(series, mode=None):
        """ Permet de lancer le contrôle du type Date sur toute la serie passée en paramètre """
        if series.dtype in FeatureTypeDate.instances or pd.api.types.is_datetime64_dtype(series):
            return True

        if mode == FeatureType.mode_convert:
            try:
                _ = [ parser.parse(f"{x}") for x in series ]
            except:
                #Si le cas n'est pas possible alors la feature n'est pas considéré comme numérique
                return False
            return True
        return False

class FeatureTypeMixed:
    tag = "mixed"
    intances = []

    @staticmethod
    def check(series, mode=None):
        pass

class FeatureType:
    # Mode où le typage est vérifié de façon très stricte, par exemple "2" sera jamais considéré comme un entier.
    mode_strict = "strict"
    # Mode où le typage est vérifié de façon plus ouverte, par exemple "2" sera considéré comme un entier.
    mode_convert = "convert"

    # Liste des contrôles de typage
    controls = [
        FeatureTypeNumerical,
        FeatureTypeBoolean,
        FeatureTypeURL,
        FeatureTypeDate,
        FeatureTypePath
    ]

    def __init__(self, df, mode=None):
        self.mode = mode or FeatureType.mode_strict
        self.df = df.copy()

    def check_all(self):
        """ Vérification du type pour chaque feature du dataset """
        features = {}
        for c in self.df.columns:

            tags = []
            for control_instance in FeatureType.controls:
                if control_instance.check(self.df[c], self.mode):
                    tags.append(control_instance.tag)

            if len(tags) == 0:
                tags.append( FeatureTypeMixed.tag )

            features[c] = tags

        return features