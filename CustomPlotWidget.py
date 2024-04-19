from pyqtgraph import PlotWidget

from CustomViewBox import CustomViewBox


class CustomPlotWidget(PlotWidget):
    def __init__(self, parent=None):
        PlotWidget.__init__(self, parent, viewBox=CustomViewBox())
