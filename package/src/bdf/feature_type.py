from urllib.parse import urlparse
from dateutil import parser
import os

class FeatureExtraction:

    def __init__(self, df):

        self.list_of_chars = "abcdefghijklmnopqrstuvwxyz0123456789_"
        self.chars_to_index = {}
        self.index_to_chars = {}
        self.vector_size = 25

        self.df = df
        self.columns = df.columns

        for i, v in enumerate(self.list_of_chars):
            self.chars_to_index[v] = i
            self.index_to_chars[i] = v

    def conv_to_vec(self, input):
        """ Permet de convertir l'entrée en vecteur d'une dimension """
        output = []
        for c in input:
            output.append(self.chars_to_index[c])

        if len(output) >= self.vector_size:
            return output[0:self.vector_size]
        else:
            return output + [0] * (self.vector_size - len(output))

    def run(self, series):

        features = {
            'is_null' : 0
        }
        name_to_vector = self.conv_to_vec(series.name)
        features = features | dict(zip([f"f_{x}" for x in range(len(name_to_vector))], name_to_vector))
        if series.empty or series.isnull().all():
            features['is_null'] = 1

        return features

t = pd.DataFrame(data={"user_id" : [0,1,2]})
f = FeatureExtraction(t)
f.run(t['user_id'])


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
    extension_not_allow = []

    @staticmethod
    def check(series, mode=None):
        """ Permet de lancer le contrôle du type URL sur toute la serie passée en paramètre """
        if series.dtype in FeatureTypePath.instances:
            for x in series:
                u = urlparse(f"{x}")
                if (u.netloc != "" or u.scheme != "") or (u.path != "" and (FeatureTypePath.is_file_path(u.path) == False and FeatureTypePath.is_dir_path(u.path) == False)):
                    return False
            return True

        return False

    @staticmethod
    def is_file_path(p):
        """ Permet de vérifier si la chaine de caractères passée en paramètre correspond à un path de fichier """

        s = os.path.split(p)
        #Si présence d'un point dans le positionnement du fichier en théorie dans le path alors il s'agit d'un path de chemin et non d'un path de dossier
        if s[1] != "":
            if "." in s[1]:
                extension = s[1].split('.')
                return extension[-1] not in FeatureTypePath.extension_not_allow

        return False

    @staticmethod
    def is_dir_path(p):
        """ Permet de verifier si la chaine de caractères passée en paramètre correspond à un path de dossier sans verifier son existence """

        #Si aucun séparateur de dossier n'est présent alors on ne cherche pas plus loin, il ne s'agit pas d'un path de dossier
        if not any([ sep in p for sep in ['/', '\\'] ]):
            return False

        s = os.path.split(p)
        #Si présence d'un point dans le positionnement du fichier en théorie dans le path alors il s'agit d'un path de chemin et non d'un path de dossier
        if s[1] != "":
            if "." in s[1]:
                return False

        return True

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

class FeatureTypeId:
    tag = "ID"
    instances = []

    @staticmethod
    def check(series, mode=None):
        """ Permet de lancer le contrôle du type ID sur toute la série passée en paramètre """
        # Si le nom de la feature comporte le terme ID alors on tente de confirmer qu'il s'agisse bien d'une feature de type ID
        if '_id' in series.name.lower():
            return True

        # nu = series.nunique()
        # n = len(series)
        # ratio = nu * 100 / n
        # # On par du principe que si plus de 98% des valeurs sont différentes au sein de la serie alors il s'agit d'un ID
        # print(series.name, ratio)
        # if ratio > 98.0:
        #     return True

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
        FeatureTypePath,
        FeatureTypeId,
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


aa = {
    "user_id" : [0,1,2,3,4,5,6],
    "a-a" : ["0.2",1,2,"3",4,5,6.0],
    "b" : [0,1,'0',True,0,"1",0],
    "c" : [True, False, "True", False, True, "False", True],
    "d" : ['2012-12-13 00:00:00', '2012-12-14 00:00:00', '2012-12-15 00:00:00', '2012-12-16 00:00:00', '2012-12-17 00:00:00', '2012-12-18 00:00:00', '2012-12-19 00:00:00'],
    "e" : ['/home/p.csv', '/home/p.csv', '/home/p.csv', '/home/p.csv', '/home/p.csv', '/home/p.csv', '/home/p.csv'],
    "f" : ['p.csv', 'p.csv', 'p.csv', 'p.csv', 'p.csv', 'p.csv', 'p.csv'],
    "gidout" : ['kjdfk/llksd/', 'kjdfk/llksd/','kjdfk/llksd/','kjdfk/llksd/','kjdfk/llksd/','kjdfk/llksd/','kjdfk/llksd/'],
}
df_test = pd.DataFrame(aa)

# # print(df.columns)
# all_types = {}
# for c in df_test.columns:
#     all_types[c] = FeatureType.check(df_test[c])

# all_types

ft = FeatureType(df, mode=FeatureType.mode_convert)
ft.check_all()