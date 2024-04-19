from PyQt6 import QtWidgets


class ChkBxFileDialog(QtWidgets.QFileDialog):
    def __init__(self, chkBxTitle="", filter="*.txt"):
        super().__init__(caption="Сохранить как", filter=filter)
        self.setSupportedSchemes(["file"])
        self.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        self.selectNameFilter("*.txt")
        self.chkBx = QtWidgets.QCheckBox(chkBxTitle)
        self.layout().addWidget(self.chkBx)