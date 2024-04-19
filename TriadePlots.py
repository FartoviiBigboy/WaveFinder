import datetime
import pyqtgraph
from pyqtgraph import DateAxisItem


class TriadePlots:
    def __init__(self, plots):
        self.plots = list(plots)
        self.plotItems = []
        self.crosshair_updaters = []
        self.base_pen = (128, 128, 128)

    def get_utc_local_diff_hours(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        local_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=0)))
        return (local_now - now).total_seconds() / 3600

    def plot(self, x, ys):
        self.plotItems = []
        self.plotItems.extend(plot.plot(x, y, pen=self.base_pen) for plot, y in zip(self.plots, ys))
        for plot in self.plots:
            plot.plotItem.setAxisItems({'bottom': DateAxisItem(utcOffset=self.get_utc_local_diff_hours())})
            plot.setBackground('w')
            plot.getAxis('bottom').setTextPen('k')
            plot.getAxis('left').setTextPen('k')
            plot.plotItem.getAxis('left').setWidth(50)
            plot.setLimits(xMin=x[0], xMax=x[len(x) - 1])

    def addItems(self, items):
        for plot, item in zip(self.plots, items):
            plot.addItem(item, ignoreBounds=True)

    def setMouseMovedUpdaters(self, method):
        self.crosshair_updaters.extend(
            pyqtgraph.SignalProxy(plot.scene().sigMouseMoved, rateLimit=60, slot=method)
            for plot in self.plots)

    def setDatas(self, x, ys):
        for plotItem, y in zip(self.plotItems, ys):
            plotItem.setData(x, y)

    def removeItems(self, items):
        for plot, item in zip(self.plots, items):
            plot.removeItem(item)



