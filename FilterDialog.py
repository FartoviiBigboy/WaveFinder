from PyQt6 import QtCore
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtGui import QIntValidator
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import QDialogButtonBox

from resources.ui_FilterDialog import Ui_Dialog


class FilterDialog(QDialog):
    def __init__(self, parent):
        QDialog.__init__(self, parent)

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.dialog_btnbox.button(QDialogButtonBox.StandardButton.Ok).setText("Применить")
        self.ui.dialog_btnbox.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")

        self.button_group = self.__initialize_button_group()
        self.type_by_button = self.__initialize_type_by_buttons()
        self.filter_type = "lowpass"

        self.__initialize_validation()

        self.ui.dialog_btnbox.accepted.connect(self.accept)
        self.ui.dialog_btnbox.rejected.connect(self.reject)
        self.button_group.buttonClicked.connect(self._on_radio_button_clicked)

        self._on_radio_button_clicked(self.ui.radioButton)

    def _on_radio_button_clicked(self, button):
        self.filter_type = self.type_by_button[button]
        if self.filter_type in ["bandpass", "bandstop"]:
            self.__change_input_view("F_low (Hz)", True)
        else:
            self.__change_input_view("F (Hz)", False)

    def __change_input_view(self, low_label, visibility):
        self.ui.frequency_low_label.setText(low_label)
        self.ui.frequency_high_label.setVisible(visibility)
        self.ui.frequency_high_edit.setVisible(visibility)

    def __initialize_button_group(self):
        button_group = QButtonGroup()
        button_group.addButton(self.ui.radioButton)
        button_group.addButton(self.ui.radioButton_2)
        button_group.addButton(self.ui.radioButton_3)
        button_group.addButton(self.ui.radioButton_4)
        return button_group

    def __initialize_type_by_buttons(self):
        return {
            self.ui.radioButton: "lowpass",
            self.ui.radioButton_2: "highpass",
            self.ui.radioButton_3: "bandpass",
            self.ui.radioButton_4: "bandstop"
        }

    def __initialize_validation(self):
        double_val = QDoubleValidator(0.0, 100.0, 2)
        double_val.setLocale(QtCore.QLocale("en_US"))
        int_val = QIntValidator(0, 100)
        int_val.setLocale(QtCore.QLocale("en_US"))
        self.ui.filter_order_edit.setValidator(int_val)
        self.ui.frequency_low_edit.setValidator(double_val)
        self.ui.frequency_high_edit.setValidator(double_val)
