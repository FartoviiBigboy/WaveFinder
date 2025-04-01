import copy

from Seismogram import Seismogram
import nptyping as npt
import scipy
import numpy as np


class PredictionFilter:
    WAVE_LENGTH: int = 400
    NUMBER_OF_TRACES: int = 3
    DELTA_X: int = 40
    BATCH_SIZE: int = 32

    EPS: float = 0.000001
    MAX_WINDOWS: int = 50

    @staticmethod
    def s_wave_correction(
            seismogram: Seismogram,
            p_der_indexes: list[int],
            prediction: npt.NDArray[npt.Shape["*, 3"], npt.Float32]
    ) -> npt.NDArray[npt.Shape["*, 3"], npt.Float32]:
        copy_prediction = copy.deepcopy(prediction)
        orig_trace = seismogram.get_original_interpolated()
        filtered_trace = seismogram.traces
        # print("Original trace shape: ", orig_trace.shape)
        sos_low_filter = scipy.signal.butter(2, [0.1, 5], "bandpass", fs=Seismogram.NN_sampling_rate, output='sos')
        sos_high_filter = scipy.signal.butter(2, [1, 10], "bandpass", fs=Seismogram.NN_sampling_rate, output='sos')

        for i, index in enumerate(p_der_indexes):

            max_windows = min(prediction.shape[0], PredictionFilter.MAX_WINDOWS)
            if i + 1 < len(p_der_indexes) and p_der_indexes[i + 1] - p_der_indexes[i] < max_windows:
                max_windows = max(0, p_der_indexes[i + 1] - p_der_indexes[i] - 2)

            right_index = index + 1 + max_windows - 1
            if right_index >= prediction.shape[0]:
                right_index = prediction.shape[0]

            for j in range(index + 1, right_index):
                noise_index = index - PredictionFilter.WAVE_LENGTH / PredictionFilter.DELTA_X / 2

                left_noise = max(0, int(noise_index * PredictionFilter.DELTA_X - PredictionFilter.WAVE_LENGTH / 2))
                right_noise = min(orig_trace.shape[1], int(noise_index * PredictionFilter.DELTA_X + PredictionFilter.WAVE_LENGTH / 2))
                if left_noise >= right_noise:
                    continue

                left_current = max(0, int(j * PredictionFilter.DELTA_X - PredictionFilter.WAVE_LENGTH / 2))
                right_current = min(orig_trace.shape[1], int(j * PredictionFilter.DELTA_X + PredictionFilter.WAVE_LENGTH / 2))
                if left_current >= right_current:
                    continue

                noise_data = orig_trace[:, left_noise: right_noise]
                current_data = orig_trace[:, left_current: right_current]

                noise_data_ampl = filtered_trace[:, left_noise: right_noise]
                current_data_ampl = filtered_trace[:, left_current: right_current]

                amplitude_noise = PredictionFilter.__RMS3(noise_data_ampl)
                amplitude_current = PredictionFilter.__RMS3(current_data_ampl)
                relative_amplitude = amplitude_current / amplitude_noise

                low_noise = PredictionFilter.__RMS3(scipy.signal.sosfilt(sos_low_filter, noise_data))
                high_noise = PredictionFilter.__RMS3(scipy.signal.sosfilt(sos_high_filter, noise_data))
                energy_noise = low_noise / (high_noise + PredictionFilter.EPS)

                low_current = PredictionFilter.__RMS3(scipy.signal.sosfilt(sos_low_filter, current_data))
                high_current = PredictionFilter.__RMS3(scipy.signal.sosfilt(sos_high_filter, current_data))
                energy_current = low_current / (high_current + PredictionFilter.EPS)

                relative_energy = energy_current / energy_noise

                # multiplying_amplitude = 1
                # if relative_amplitude >= 1.5:
                #     multiplying_amplitude = 1.2
                # elif relative_amplitude >= 3:
                #     multiplying_amplitude = 1.5

                multiplying_amplitude = 1 + 0.5 / (1 + np.exp(-2 * (relative_amplitude - 1.5)))

                # multiplying_energy = 1
                # if relative_energy >= 2:
                #     multiplying_energy = 1.2

                multiplying_energy = 1 + 0.2 / (1 + np.exp(-2 * (relative_energy - 2)))

                coeff = multiplying_amplitude * multiplying_energy
                # print(f"mul ampl {multiplying_amplitude} \n"
                #       f"mul en {multiplying_energy} \n"
                #       f"for i - {i} ({index * PredictionFilter.DELTA_X}) and j - {j}")

                copy_prediction[j, 1] *= coeff

        return copy_prediction

    @staticmethod
    def __RMS3(window: npt.NDArray[npt.Shape["3, *"], npt.Float32]) -> float:
        rms_value_n = np.sqrt(np.mean(np.square(window[0])))
        rms_value_e = np.sqrt(np.mean(np.square(window[1])))
        rms_value_z = np.sqrt(np.mean(np.square(window[2])))
        return (rms_value_n + rms_value_e + rms_value_z) / 3.0
