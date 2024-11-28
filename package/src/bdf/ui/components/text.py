
from bdf.ui.component import Component

class TextBlock(Component):
    """ Classe représentant le composant Text Block UI """

    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self.tag = 'p'

class TextInline(Component):
    """ Classe représentant le composant Text Inline UI """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'span'

