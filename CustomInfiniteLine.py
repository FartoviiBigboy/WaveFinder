from pyqtgraph import InfiniteLine


class CustomInfiniteLine(InfiniteLine):
    def __init__(self, parent=None, **kwargs):
        self.parent = parent
        super().__init__(**kwargs)
