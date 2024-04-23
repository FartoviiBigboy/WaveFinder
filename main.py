import os
import sys

import obspy
import PyQt6
from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QAbstractItemView
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QListWidgetItem
from PIL import Image
from PIL import ImageQt
import pyqtgraph
import pyqtgraph.exporters

from ChkBxFileDialog import ChkBxFileDialog
from TraceWidget import TraceWidget
from Seismogram import Seismogram
from FilterDialog import FilterDialog

from resources.ui_MainWindow import Ui_MainForm
from NeuralNetworkModel import NeuralNetworkModel


class MainWindow(QtWidgets.QWidget):

    def __init__(self, model, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.ui = Ui_MainForm()
        self.ui.setupUi(self)
        self.ui.load_seismogram_btn.clicked.connect(self.add_seismograms)
        self.ui.temp_btn.clicked.connect(self.clear_seismogram_list)
        self.ui.apply_filter_btn.clicked.connect(self.apply_filter)
        self.ui.invert_selection_btn.clicked.connect(self.invert_selection)
        self.ui.reset_filters_btn.clicked.connect(self.reset_seismograms)
        self.ui.apply_NN_btn.clicked.connect(self.apply_NN)
        self.ui.for_all_chkbox.clicked.connect(self.select_all)
        self.ui.save_results_btn.clicked.connect(self.save_seismograms)

        self.ui.seismogram_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.ui.seismogram_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.ui.seismogram_list.setAutoScroll(False)

        self.ui.progress_bar.setVisible(False)

        self.file_formats = {
            "PNG (*.png)": self.save_as_png,
            "PKS (*.pks)": self.save_as_pks,
            "RAW (*.raw)": self.save_raw
        }

        self.trace_widgets_list: list[TraceWidget] = []
        self.model = model

    @pyqtSlot()
    def select_all(self):
        for trace in self.trace_widgets_list:
            if self.ui.for_all_chkbox.isChecked():
                trace.ui.apply_operation_chkbox.setChecked(True)
            else:
                trace.ui.apply_operation_chkbox.setChecked(False)

    @pyqtSlot()
    def apply_NN(self):
        for trace in self.trace_widgets_list:
            if trace.ui.apply_operation_chkbox.isChecked():
                prediction, p_der_indexes, s_der_indexes = self.model.get_prediction(trace.seismogram,
                                                                                     self.ui.progress_bar)
                trace.set_prediction(prediction, p_der_indexes, s_der_indexes)
                trace.enable_sliders()

    @pyqtSlot()
    def reset_seismograms(self):
        for trace in self.trace_widgets_list:
            if trace.ui.apply_operation_chkbox.isChecked():
                trace.seismogram.reset_trace()
                trace.reset_prediction()
                trace.refresh_plots()
                trace.reset_sliders()

    @pyqtSlot()
    def invert_selection(self):
        for trace in self.trace_widgets_list:
            if trace.ui.apply_operation_chkbox.isChecked():
                trace.ui.apply_operation_chkbox.setChecked(False)
            else:
                trace.ui.apply_operation_chkbox.setChecked(True)

    @pyqtSlot()
    def apply_filter(self):
        dlg = FilterDialog(self)
        if dlg.exec():
            order = int(dlg.ui.filter_order_edit.text())
            filter_type = dlg.filter_type
            if filter_type in ["bandpass", "bandstop"]:
                frequency = [float(dlg.ui.frequency_low_edit.text()), float(dlg.ui.frequency_high_edit.text())]
            else:
                frequency = float(dlg.ui.frequency_low_edit.text())
            for trace in self.trace_widgets_list:
                if trace.ui.apply_operation_chkbox.isChecked():
                    try:
                        trace.seismogram.apply_filter(order, frequency, filter_type)
                        trace.refresh_plots()
                    except Exception:
                        QMessageBox.critical(self, "Ошибка примения фильтрации",
                                             "Заданы некорректные параметры фильтра!")
                        break

    @pyqtSlot()
    def add_seismograms(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, 'Открыть файл(ы) сейсмограмм(ы)', '',
                                                     'MSEED files (*.mseed);;All Files (*)')
        traces = []
        if file_paths:
            for file in file_paths:
                try:
                    sts = self.sort_stations(obspy.read(file))
                    traces.extend([Seismogram(sts[i:i + 3], file)
                                   for i in range(0, len(sts), 3)])
                except Exception:
                    QMessageBox.critical(
                        self,
                        "Ошибка чтения файла",
                        f"файл {file} поврежден или не поддерживается!",
                    )

        for trace in traces:
            wdt = TraceWidget(trace)
            my_item = QListWidgetItem(self.ui.seismogram_list)
            my_item.setSizeHint(wdt.size())
            self.trace_widgets_list.append(wdt)
            self.ui.seismogram_list.addItem(my_item)
            self.ui.seismogram_list.setItemWidget(my_item, wdt)

        traces.clear()

    @pyqtSlot()
    def save_seismograms(self):
        files_types = "PNG (*.png);;PKS (*.pks);; RAW (*.raw)"
        dialog = ChkBxFileDialog(chkBxTitle="Сохранить как последовательность", filter=files_types)

        for i in range(len(self.trace_widgets_list)):
            if self.trace_widgets_list[i].ui.apply_operation_chkbox.isChecked() and \
                    dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.chkBx.isChecked():
                    for j in range(i, len(self.trace_widgets_list)):
                        self.file_formats[dialog.selectedNameFilter()](self.trace_widgets_list[j], dialog, True)
                    break
                else:
                    self.file_formats[dialog.selectedNameFilter()](self.trace_widgets_list[i], dialog, False)

    def save_raw(self, trace, dlg, is_long_name):
        if trace.ui.apply_operation_chkbox.isChecked():
            file_path = f"{self.get_file_path(trace, dlg, is_long_name)}.raw"
            raw_data = trace.get_raw()
            with open(file_path, 'w') as file:
                for item in raw_data:
                    file.write(item + '\n')

    def save_as_pks(self, trace, dlg, is_long_name):
        if trace.ui.apply_operation_chkbox.isChecked():
            file_path = f"{self.get_file_path(trace, dlg, is_long_name)}.pks"
            log_file = trace.get_lines_as_pks()
            with open(file_path, 'w') as file:
                for item in log_file:
                    file.write(item + '\n')

    def save_as_png(self, trace, dlg, is_long_name):
        if trace.ui.apply_operation_chkbox.isChecked():
            file_path = f"{self.get_file_path(trace, dlg, is_long_name)}.png"
            image = self.get_complex_image(trace)
            image.save(file_path)

    # TODO
    # file_path = f"{dlg.selectedUrls()[0].toLocalFile()}_{os.path.basename(trace.seismogram.file_path).split('.')[0]}"
    def get_file_path(self, trace, dlg, is_long_name):
        if is_long_name:
            file_path = f"{dlg.selectedUrls()[0].toLocalFile()}_{os.path.basename(trace.seismogram.file_path)}_{trace.seismogram.station_name}"
        else:
            file_path = f"{dlg.selectedUrls()[0].toLocalFile()}"
        return file_path

    def get_complex_image(self, trace):
        images = [ImageQt.fromqimage(pyqtgraph.exporters.ImageExporter(x).export(toBytes=True))
                  for x in [trace.ui.N_trace.scene(), trace.ui.E_trace.scene(), trace.ui.Z_trace.scene()]]
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)

        new_im = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        return new_im

    # TODO
    # first_value = self.trace_widgets_list[0].ui.p_sensitivity.value()
    # second_value = self.trace_widgets_list[0].ui.noise_p_sensitivity.value()
    # third_value = self.trace_widgets_list[0].ui.s_sensitivity.value()
    # fourth_value = self.trace_widgets_list[0].ui.noise_s_sensitivity.value()
    # for tr_wdt in self.trace_widgets_list:
    #     tr_wdt.object_to_types[PVerticalTriadeLine]["val"].setValue(first_value)
    #     tr_wdt.object_to_types[PVerticalTriadeLine]["noise"].setValue(second_value)
    #     tr_wdt.object_to_types[SVerticalTriadeLine]["val"].setValue(third_value)
    #     tr_wdt.object_to_types[SVerticalTriadeLine]["noise"].setValue(fourth_value)

    @pyqtSlot()
    def clear_seismogram_list(self):
        self.ui.seismogram_list.clear()
        self.trace_widgets_list.clear()

    def keyPressEvent(self, event):
        if event.key() == PyQt6.QtCore.Qt.Key.Key_Delete and len(self.ui.seismogram_list.selectedItems()) > 0:
            list_items = self.ui.seismogram_list.selectedItems()
            for item in list_items:
                widget = self.ui.seismogram_list.itemWidget(item)
                self.trace_widgets_list.remove(widget)
                self.ui.seismogram_list.takeItem(self.ui.seismogram_list.row(item))
        event.accept()

    def sort_stations(self, st):
        sorted_list = sorted(st, key=lambda x: (x.stats.station, x.stats.channel))
        sorted_list[::3], sorted_list[1::3] = sorted_list[1::3], sorted_list[::3]
        return sorted_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    network_model = NeuralNetworkModel()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(network_model)
    window.show()
    app.exec()
