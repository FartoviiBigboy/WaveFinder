from pyqtgraph import InfiniteLine


class CustomInfiniteLine(InfiniteLine):
    def __init__(self, parent=None, **kwargs) -> None:
        self.parent = parent
        super().__init__(**kwargs)
