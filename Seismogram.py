import numpy as np
import obspy
import scipy


class Seismogram:
    NN_sampling_rate = 100

    def __init__(self, traces, file_path):
        self.station_name = traces[0].stats.station
        self.sampling_rate = traces[0].stats.sampling_rate
        self.network = traces[0].stats.network
        self.channels = [traces[0].stats.channel, traces[1].stats.channel, traces[2].stats.channel]
        self.start_time, self.end_time = obspy.io.mseed.util.get_start_and_end_time(file_path)

        self.file_path = file_path
        self.original_traces = traces.copy()
        self.traces = self.__interpolate_traces()

    def __interpolate_traces(self):
        if self.sampling_rate == Seismogram.NN_sampling_rate:
            return np.array(self.original_traces)
        else:
            return np.array(
                [self.original_traces[0].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate),
                 self.original_traces[1].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate),
                 self.original_traces[2].copy().interpolate(sampling_rate=Seismogram.NN_sampling_rate)])

    def reset_trace(self):
        self.traces = self.__interpolate_traces()

    def apply_filter(self, order: int, frequency, filter_type: str):
        sos = scipy.signal.butter(order, frequency, filter_type, fs=Seismogram.NN_sampling_rate, output='sos')
        self.traces = scipy.signal.sosfilt(sos, self.traces)

