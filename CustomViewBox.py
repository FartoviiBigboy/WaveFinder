from PyQt6.QtCore import Qt
from pyqtgraph import ViewBox


class CustomViewBox(ViewBox):

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.ignore()
        else:
            ViewBox.mouseDragEvent(self, ev)
