
from bdf.ui.component import Component

class Container(Component):
    """ Classe représentant le composant Container UI """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.tag = 'div'

class Html(Component):
    """ Classe représentant le composant Html UI """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'html'

class Head(Component):
    """ Classe représentant le composant Head UI """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'head'

class Body(Component):
    """ Classe représentant le composant Body UI """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'body'






