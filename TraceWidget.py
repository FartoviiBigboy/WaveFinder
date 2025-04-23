import contextlib
import os
import datetime
import threading
from collections.abc import Callable
from itertools import chain

import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSlot
from obspy import UTCDateTime
from tensorflow.python.ops.gen_array_ops import deep_copy

from Seismogram import Seismogram

from resources.ui_TraceWidget import Ui_TraceWidget
from NeuralNetworkModel import NeuralNetworkModel

from TriadePlots import TriadePlots
from TriadeLines import VerticalTriadeLine, PVerticalTriadeLine, SVerticalTriadeLine
from PredictionFilter import PredictionFilter


class TraceWidget(QtWidgets.QWidget):
    def __init__(self, seismogram: Seismogram, parent=None) -> None:
        QtWidgets.QWidget.__init__(self, parent)

        self.update_mutex: threading.Lock = threading.Lock()
        self.selected_item = None

        self.prediction = None
        self.p_der_indexes = None
        self.s_der_indexes = None

        self.p_filtered_indexes: list[int] = []
        self.s_filtered_indexes: list[int] = []

        self.p_vertical_lines = []
        self.s_vertical_lines = []

        self.seismogram: Seismogram = seismogram
        self.ui: Ui_TraceWidget = Ui_TraceWidget()
        self.ui.setupUi(self)
        self.ui.file_label.setText(os.path.basename(self.seismogram.file_path))
        self.ui.station_label.setText(self.seismogram.station_name)

        self.p_types = {}
        self.s_types = {}

        self.ui_sliders: list[QtWidgets.QSlider] = self._initialize_ui_sliders()
        self.ui_label_sliders: dict[QtWidgets.QSlider, QtWidgets.QLabel] = self._initialize_ui_label_sliders()
        self.object_to_types = self._initialize_object_to_types()
        self.key_to_method: dict[QtCore.Qt.Key, Callable] = self._initialize_key_to_method()
        self.crosshair_vertical_lines: VerticalTriadeLine = VerticalTriadeLine()
        self.timestamp_list: list[float] = self._generate_timestamp_list()
        self.trace_plots: TriadePlots = self._initialize_plots()

    def _initialize_ui_sliders(self):
        return [self.ui.p_sensitivity, self.ui.noise_p_sensitivity, self.ui.s_sensitivity, self.ui.noise_s_sensitivity]

    def _initialize_ui_label_sliders(self) -> dict[QtWidgets.QSlider, QtWidgets.QLabel]:
        return {
            self.ui.p_sensitivity: self.ui.p_value_label,
            self.ui.noise_p_sensitivity: self.ui.noise_p_value_label,
            self.ui.s_sensitivity: self.ui.s_value_label,
            self.ui.noise_s_sensitivity: self.ui.noise_s_value_label
        }

    def _initialize_object_to_types(self):
        return {
            PVerticalTriadeLine: self.p_types,
            self.ui.p_sensitivity: self.p_types,
            self.ui.noise_p_sensitivity: self.p_types,
            SVerticalTriadeLine: self.s_types,
            self.ui.s_sensitivity: self.s_types,
            self.ui.noise_s_sensitivity: self.s_types
        }

    def _initialize_key_to_method(self) -> dict[QtCore.Qt.Key, Callable]:
        return {
            QtCore.Qt.Key.Key_D: self._delete_line,
            QtCore.Qt.Key.Key_R: self._remove_line,
            QtCore.Qt.Key.Key_M: self._switch_movable_line,
            QtCore.Qt.Key.Key_F: self._add_line_p,
            QtCore.Qt.Key.Key_B: self._add_line_s,
            QtCore.Qt.Key.Key_N: self._normalize_y_range,
        }

    def _generate_timestamp_list(self) -> list[float]:
        initial_date: float = self.seismogram.start_time.timestamp
        final_date: float = self.seismogram.end_time.timestamp
        time_diff: float = final_date - initial_date
        interval: float = time_diff / (len(self.seismogram.traces[0]) - 1)
        return [
            initial_date + i * interval
            for i in range(len(self.seismogram.traces[0]))
        ]

    def _initialize_plots(self) -> TriadePlots:
        plots: TriadePlots = TriadePlots([self.ui.N_trace, self.ui.E_trace, self.ui.Z_trace])
        plots.plot(self.timestamp_list, self.seismogram.traces)
        plots.addItems(self.crosshair_vertical_lines.lines)
        plots.setMouseMovedUpdaters(self._update_crosshair)
        self.ui.E_trace.setXLink(self.ui.Z_trace)
        self.ui.N_trace.setXLink(self.ui.Z_trace)
        return plots

    def _update_crosshair(self, event) -> None:
        coordinates = event[0]
        for plot in self.trace_plots.plots:
            if plot.sceneBoundingRect().contains(coordinates):
                mouse_point = plot.plotItem.vb.mapSceneToView(coordinates)
                self.ui.time_on_graph.setText(str(self._converted_time_from_timestamp(mouse_point.x())))
                self.crosshair_vertical_lines.setPos(mouse_point.x())
            break

    def _converted_time_from_timestamp(self, timestamp):
        return datetime.datetime.utcfromtimestamp(timestamp)

    def set_prediction(self, prediction: np.ndarray, p_der_indexes: list[int], s_der_indexes: list[int]) -> None:
        self.prediction = prediction
        self.p_der_indexes = p_der_indexes
        self.s_der_indexes = s_der_indexes

        self.p_types.update(self._get_p_types())
        self.s_types.update(self._get_s_types())

    def _get_p_types(self):
        return {
            "name": "P",
            "val": self.ui.p_sensitivity,
            "noise": self.ui.noise_p_sensitivity,
            "pred": self.prediction[:, 0],
            "der": self.p_der_indexes,
            "filt": self.p_filtered_indexes,
            "lines": self.p_vertical_lines,
            "class_line": PVerticalTriadeLine
        }

    def _get_s_types(self):
        return {
            "name": "S",
            "val": self.ui.s_sensitivity,
            "noise": self.ui.noise_s_sensitivity,
            "pred": self.prediction[:, 1],
            "pred_orig": self.prediction[:, 1],
            "der": self.s_der_indexes,
            "filt": self.s_filtered_indexes,
            "lines": self.s_vertical_lines,
            "class_line": SVerticalTriadeLine
        }

    def refresh_plots(self) -> None:
        self.trace_plots.setDatas(self.timestamp_list, self.seismogram.traces)

    def enable_sliders(self) -> None:
        for slider, label in zip(self.ui_sliders, self.ui_label_sliders):
            slider.setEnabled(True)
            slider.valueChanged.connect(self._update_values)
            slider.valueChanged.connect(self._update_labels)
        self.ui.p_sensitivity.setValue(1)
        self.ui.s_sensitivity.setValue(1)

    # TODO update filtering
    def _update_values(self, value) -> None:
        with self.update_mutex:
            sender = self.object_to_types[self.sender()]
            sender["filt"].clear()
            first_value, second_value = self._get_thresholds(sender, self.sender(), value)
            for index in sender["der"]:
                if sender["pred"][index] > first_value and self.prediction[index, 2] < second_value:
                    sender["filt"].append(index)

            if sender["name"] == "P":
                self.s_types["pred"] = []
                self.s_types["pred"] = PredictionFilter.s_wave_correction(self.seismogram, sender["filt"],
                                                                          self.prediction)[:, 1]
            print(sender["name"])
            self._show_prediction(sender)

    def _update_labels(self, value) -> None:
        label: QtWidgets.QLabel = self.ui_label_sliders[self.sender()]
        label.setText(str(float(self.sender().value()) / 100))

    def _get_thresholds(self, sender, widget, value) -> tuple[float, float]:
        first_value: float = float(sender["val"].value()) / 100.0
        second_value: float = float(sender["noise"].value()) / 100.0
        if widget in [self.ui.p_sensitivity, self.ui.s_sensitivity]:
            first_value = float(value) / 100.0
        if widget in [self.ui.noise_p_sensitivity, self.ui.noise_s_sensitivity]:
            second_value = float(value) / 100.0
        return first_value, second_value

    def _show_prediction(self, sender) -> None:
        self._clear_prediction(sender)
        for index in sender["filt"]:
            converted_index: int = self._graphic_index_from_model(index)
            vertical_line = sender["class_line"](pos=self.timestamp_list[converted_index])
            vertical_line.connectClick(self._click_prediction)
            sender["lines"].append(vertical_line)
            self.trace_plots.addItems(vertical_line.lines)

    def _clear_prediction(self, sender) -> None:
        for triade in sender["lines"]:
            self.trace_plots.removeItems(triade.lines)
        sender["lines"].clear()

    def _graphic_index_from_model(self, index) -> int:
        return int(NeuralNetworkModel.DELTA_X * index)

    def reset_prediction(self) -> None:
        for trace in chain(self.p_vertical_lines, self.s_vertical_lines):
            self.trace_plots.removeItems(trace.lines)
        self.p_vertical_lines.clear()
        self.s_vertical_lines.clear()
        self.prediction = None
        self.p_der_indexes = None
        self.s_der_indexes = None

    def reset_sliders(self) -> None:
        for slider in self.ui_sliders:
            with contextlib.suppress(Exception):
                slider.valueChanged.disconnect()
            slider.valueChanged.connect(self._update_labels)
            slider.setEnabled(False)
        self.ui.p_sensitivity.setValue(0)
        self.ui.s_sensitivity.setValue(0)
        self.ui.noise_p_sensitivity.setValue(80)
        self.ui.noise_s_sensitivity.setValue(80)
        for slider in self.ui_sliders:
            slider.valueChanged.disconnect()

    def get_raw(self) -> list[str]:
        result: list[str] = []
        result.append("P")
        result.extend(
            [f"{self._graphic_index_from_model(index)} {self.prediction[index, 0]} {self.prediction[index, 2]}" for
             index in self.p_der_indexes])
        result.append("S")
        result.extend(
            [f"{self._graphic_index_from_model(index)} {self.prediction[index, 1]} {self.prediction[index, 2]}" for
             index in self.s_der_indexes])
        result.append("raw")
        result.extend(
            [f"{self.prediction[index, 0]} {self.prediction[index, 1]} {self.prediction[index, 2]}" for
             index in range(self.prediction.shape[0])])
        return result

    def get_lines_as_pks(self) -> list[str]:
        log_file: list[str] = []
        for elem in chain(self.p_vertical_lines, self.s_vertical_lines):
            for i, channel in enumerate(self.seismogram.channels):
                string: str = f"#T{self.seismogram.station_name} " \
                         f"{channel} {self.seismogram.network} " \
                         f"{self.object_to_types[type(elem)]['name']} ? e " \
                         f"{datetime.datetime.utcfromtimestamp(elem.pos()).strftime('%Y%m%d%H%M%S%f')} " \
                         f"{self.seismogram.traces[i][self._graphic_index_from_position(elem.pos())]}"
                log_file.append(string)
        return log_file

    # TODO
    # def get_lines_as_pks(self):
    #     log_file = []
    #     for elem in chain(self.p_vertical_lines, self.s_vertical_lines):
    #         string = f"{self.object_to_types[type(elem)]['name']} " \
    #                  f"{self._graphic_index_from_position(elem.pos())}"
    #         log_file.append(string)
    #     return log_file

    def mousePressEvent(self, event) -> None:
        for trace in chain(self.p_vertical_lines, self.s_vertical_lines):
            trace.setPen(trace.base_pen)
        self.selected_item = None
        QtWidgets.QWidget.mousePressEvent(self, event)

    def keyPressEvent(self, event) -> None:
        with contextlib.suppress(Exception):
            self.key_to_method[event.key()]()
        event.accept()

    def _delete_line(self) -> None:
        if self.selected_item:
            model_index: int = self._model_index_from_graphic(
                self._graphic_index_from_position(self.selected_item.pos()))
            if model_index in self.object_to_types[type(self.selected_item)]["filt"]:
                self.object_to_types[type(self.selected_item)]["filt"].remove(model_index)
                self.object_to_types[type(self.selected_item)]["der"].remove(model_index)
            self._remove_line()

    def _remove_line(self) -> None:
        if self.selected_item:
            self.trace_plots.removeItems(self.selected_item.lines)
            self.object_to_types[type(self.selected_item)]["lines"].remove(self.selected_item)

    def _switch_movable_line(self) -> None:
        if self.selected_item:
            if self.selected_item.movable():
                self.selected_item.setMovables(False)
                self.selected_item.disconnectPositionFinished()
            else:
                self.selected_item.setMovables(True)
                self.selected_item.connectPositionFinished(self._handle_sig_dragged)

    def _add_line_p(self) -> None:
        self._add_line(PVerticalTriadeLine)

    def _add_line_s(self) -> None:
        self._add_line(SVerticalTriadeLine)

    def _add_line(self, line_class) -> None:
        for plot in self.trace_plots.plots:
            coordinates: QtCore.QPointF = QtGui.QCursor.pos().toPointF()
            coordinates = plot.mapFromGlobal(coordinates)
            if plot.sceneBoundingRect().contains(coordinates):
                mouse_point: QtCore.QPointF = plot.plotItem.vb.mapSceneToView(coordinates)
                vertical_line: VerticalTriadeLine = line_class(pos=mouse_point.x())
                vertical_line.connectClick(self._click_prediction)
                self.object_to_types[line_class]["lines"].append(vertical_line)
                self.trace_plots.addItems(vertical_line.lines)
                break

    def _normalize_y_range(self) -> None:
        for plot, trace in zip(self.trace_plots.plots, self.seismogram.traces):
            x_left_index: int = self._graphic_index_from_position(plot.plotItem.viewRange()[0][0])
            x_right_index: int = self._graphic_index_from_position(plot.plotItem.viewRange()[0][1])
            y_min: float = min(trace[x_left_index:x_right_index])
            y_max: float = max(trace[x_left_index:x_right_index])
            plot.setYRange(y_min, y_max)

    def _model_index_from_graphic(self, index: int) -> int:
        return int(
            (index) / NeuralNetworkModel.DELTA_X
        )

    def _graphic_index_from_position(self, x_position: float) -> int:
        initial_date: float = self.seismogram.start_time.timestamp
        final_date: float = self.seismogram.end_time.timestamp
        absolute_shift: float = x_position - initial_date
        time_diff: float = final_date - initial_date
        index: int = int(absolute_shift / time_diff * float(len(self.seismogram.traces[0])))
        if index < 0:
            index = 0
        elif index >= len(self.seismogram.traces[0]):
            index = len(self.seismogram.traces[0]) - 1
        return index

    def _click_prediction(self, ev) -> None:
        wdt = self.sender().parent
        for trace in self.object_to_types[type(wdt)]["lines"]:
            trace.setPen(trace.base_pen)
        wdt.setPen(wdt.hover_pen)
        self.selected_item = wdt

    def _handle_sig_dragged(self) -> None:
        wdt = self.sender().parent
        for plot in self.trace_plots.plots:
            coordinates = QtGui.QCursor.pos().toPointF()
            coordinates = plot.mapFromGlobal(coordinates)
            if plot.sceneBoundingRect().contains(coordinates):
                mouse_point = plot.plotItem.vb.mapSceneToView(coordinates)
                wdt.setPos(mouse_point.x())
                break
