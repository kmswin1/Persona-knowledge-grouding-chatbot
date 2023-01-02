class MarkovNode:

    def __init__(self, id, repr, multi=1.0):
        self.id = id
        self.repr = repr
        self.multi = multi
        self.children = {}