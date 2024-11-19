from bdf.dataset import Dataset
from bdf.tools import Tools

def test_instance():

    d = Dataset("./tests/fichier.csv", options={'verbose' : 1})
    print(d)

    p = Tools.random_id(pattern=['123-', '%s', '-', '%s', '%S', '%S', '%d', '%x'])
    print(p)
