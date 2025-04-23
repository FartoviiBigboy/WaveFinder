from collections.abc import Iterable

import numpy as np
import obspy
import scipy


class Seismogram:
    NN_sampling_rate: int = 100

    def __init__(self, traces: list[obspy.Trace], file_path: str) -> None:
        self.station_name: str = traces[0].stats.station
        self.sampling_rate: int = traces[0].stats.sampling_rate
        self.network: str = traces[0].stats.network
        self.channels: list[str] = [traces[0].stats.channel, traces[1].stats.channel, traces[2].stats.channel]
        self.start_time: obspy.UTCDateTime
        self.end_time: obspy.UTCDateTime
        self.start_time, self.end_time = obspy.io.mseed.util.get_start_and_end_time(file_path)

        self.file_path: str = file_path
        self.original_traces: list[obspy.Trace] = traces.copy()
        self.traces: np.ndarray = self.__interpolate_traces()

    def __interpolate_traces(self) -> np.ndarray:
        if self.sampling_rate == Seismogram.NN_sampling_rate:
            return np.array(self.original_traces)
        else:
            return np.array(
                [self.original_traces[0].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate),
                 self.original_traces[1].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate),
                 self.original_traces[2].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate)])

    def get_original_interpolated(self) -> np.ndarray:
        if self.sampling_rate == Seismogram.NN_sampling_rate:
            return np.array(self.original_traces)
        else:
            return np.array(
                [self.original_traces[0].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate),
                 self.original_traces[1].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate),
                 self.original_traces[2].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate)])

    def reset_trace(self) -> None:
        self.traces = self.__interpolate_traces()

    def apply_filter(self, order: int, frequency: float | list[float], filter_type: str) -> None:
        sos: np.ndarray | Iterable | int | float = scipy.signal.butter(order, frequency, filter_type, fs=Seismogram.NN_sampling_rate, output='sos')
        self.traces = scipy.signal.sosfilt(sos, self.traces)

