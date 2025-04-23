from itertools import accumulate

import numpy as np
import scipy
import tensorflow as tf
from PyQt6.QtWidgets import QProgressBar
import nptyping as npt

from tensorflow import keras
from keras import regularizers
from kapre import STFT, Magnitude, MagnitudeToDecibel

from Seismogram import Seismogram


class MaxABSScaler(keras.layers.Layer):
    """
    Rescale to [-1,1]
    """

    def __init__(self) -> None:
        super(MaxABSScaler, self).__init__()

    def call(self, inputs):
        min_abs_val = tf.abs(tf.reduce_min(inputs))
        max_abs_val = tf.abs(tf.reduce_max(inputs))
        max_abs = tf.maximum(min_abs_val, max_abs_val)
        return inputs / max_abs


class NeuralNetworkModel:
    WAVE_LENGTH: int = 400
    NUMBER_OF_TRACES: int = 3
    DELTA_X: int = 40
    BATCH_SIZE: int = 32

    EPS: float = 0.000001
    MAX_WINDOWS: int = 50

    def __init__(self) -> None:
        self.device_for_calculation: str = "/GPU:0" if (
                len(tf.config.list_physical_devices('GPU')) > 0) else "/device:CPU:0"
        print(self.device_for_calculation)
        self.model: tf.keras.models.Model = self.__initialize_model()
        self.__load_model_weights()

    def __load_model_weights(self) -> None:
        self.model.load_weights("resources/mymodel_3_15.h5")

    def get_prediction(self, seismogram: Seismogram, progress_bar: QProgressBar) -> tuple[
        npt.NDArray[npt.Shape["*, 3"], npt.Float32], list[int], list[int]
    ]:
        traces_copy: npt.NDArray[npt.Shape["3, *"], npt.Float32] = np.array(
            [
                np.concatenate(
                    (np.zeros(int(self.WAVE_LENGTH / 2)), seismogram.traces[0], np.zeros(int(self.WAVE_LENGTH / 2)))),
                np.concatenate(
                    (np.zeros(int(self.WAVE_LENGTH / 2)), seismogram.traces[1], np.zeros(int(self.WAVE_LENGTH / 2)))),
                np.concatenate(
                    (np.zeros(int(self.WAVE_LENGTH / 2)), seismogram.traces[2], np.zeros(int(self.WAVE_LENGTH / 2)))),
            ]
        )
        converted_traces: npt.NDArray[
            npt.Shape["*, 3, 400"], npt.Float32
        ] = self.__horizontal_2D_sliding_window(
            traces_copy,
            (self.NUMBER_OF_TRACES, self.WAVE_LENGTH),
            self.DELTA_X
        )
        converted_traces: npt.NDArray[
            npt.Shape["*, 400, 3"], npt.Float32
        ] = np.transpose(converted_traces, (0, 2, 1))
        callbacks: CustomCallback = CustomCallback(progress_bar, converted_traces.shape[0])

        predicted: npt.NDArray[npt.Shape["*, 3"], npt.Float32] = self.model.predict(
            converted_traces[:],
            batch_size=NeuralNetworkModel.BATCH_SIZE,
            callbacks=[callbacks],
            verbose=0
        )
        print(predicted.shape)
        if not isinstance(predicted,
                          npt.NDArray[npt.Shape["*, 3"], npt.Float32]):  # Error is OK, pycharm analysis error
            raise TypeError("predicted is not a NDArray(-1, 3)")

        p_der_indexes: list[int] = self.__get_maximums(predicted[:, 0])
        s_der_indexes: list[int] = self.__get_maximums(predicted[:, 1])

        # predicted = self.__s_wave_correction(seismogram, p_der_indexes, predicted)

        return predicted, p_der_indexes, s_der_indexes

    def __initialize_model(self) -> tf.keras.models.Model:
        with tf.device(self.device_for_calculation):
            output_size: int = 3
            input_size: tuple[int, int] = (400, 3)

            input_layer = tf.keras.layers.Input(shape=input_size)
            x = STFT(n_fft=64,
                     window_name=None,
                     pad_end=False,
                     hop_length=16,
                     input_data_format='channels_last',
                     output_data_format='channels_last')(input_layer)
            x = Magnitude()(x)
            x = MagnitudeToDecibel()(x)
            x = MaxABSScaler()(x)

            x = tf.keras.layers.ZeroPadding2D(padding=(2, 2))(x)
            x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)

            x = self.__res_identity(x, filters=32)
            x = self.__res_identity(x, filters=32)
            x = self.__res_identity(x, filters=32)

            x = self.__res_conv(x, s=2, filters=64)
            x = self.__res_identity(x, filters=64)
            x = self.__res_identity(x, filters=64)

            x = self.__res_conv(x, s=2, filters=128)
            x = self.__res_identity(x, filters=128)
            x = self.__res_identity(x, filters=128)

            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(output_size, activation="softmax", kernel_initializer='he_normal')(x)

            model: tf.keras.models.Model = tf.keras.models.Model(inputs=input_layer, outputs=x, name='custom_Resnet34')

            fLoss: keras.losses.SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy()
            fOptimizer: keras.optimizers.Adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
            fMetric: list[keras.metrics.SparseCategoricalAccuracy] = [keras.metrics.SparseCategoricalAccuracy()]

            model.compile(
                loss=fLoss,
                optimizer=fOptimizer,
                metrics=[fMetric]
            )

            return model

    def __res_identity(self, x, filters):
        x_skip = x

        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_regularizer=regularizers.L2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_regularizer=regularizers.L2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def __res_conv(self, x, s, filters):
        x_skip = x

        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(s, s), padding='same',
                                   kernel_regularizer=regularizers.L2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_regularizer=regularizers.L2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x_skip = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                        kernel_regularizer=regularizers.L2(0.001))(x_skip)
        x_skip = tf.keras.layers.BatchNormalization()(x_skip)

        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation("relu")(x)

        return x

    # def __get_maximums(self, array):
    #     return [i
    #             for i in range(1, len(array) - 1)
    #             if (array[i] - array[i - 1] > array[i + 1] - array[i])]

    def __get_maximums(self, array) -> list[int]:
        return [i
                for i in range(1, len(array) - 1)
                if (array[i] - array[i - 1] > 0 >= array[i + 1] - array[i])]

    # def __s_wave_correction(
    #         self,
    #         seismogram: Seismogram,
    #         p_der_indexes: list[int],
    #         prediction: npt.NDArray[npt.Shape["*, 3"], npt.Float32]
    # ) -> npt.NDArray[npt.Shape["*, 3"], npt.Float32]:
    #
    #     orig_trace = seismogram.get_original_interpolated()
    #     filtered_trace = seismogram.traces
    #     # print("Original trace shape: ", orig_trace.shape)
    #     sos_low_filter = scipy.signal.butter(2, [0.1, 5], "bandpass", fs=Seismogram.NN_sampling_rate, output='sos')
    #     sos_high_filter = scipy.signal.butter(2, [1, 10], "bandpass", fs=Seismogram.NN_sampling_rate, output='sos')
    #
    #     for i, index in enumerate(p_der_indexes):
    #
    #         max_windows = min(prediction.shape[0], self.MAX_WINDOWS)
    #         if i + 1 < len(p_der_indexes) and p_der_indexes[i + 1] - p_der_indexes[i] < max_windows:
    #             max_windows = max(0, p_der_indexes[i + 1] - p_der_indexes[i] - 2)
    #
    #         right_index = index + 1 + max_windows - 1
    #         if right_index >= prediction.shape[0]:
    #             right_index = prediction.shape[0]
    #
    #         for j in range(index + 1, right_index):
    #             noise_index = index - self.WAVE_LENGTH / self.DELTA_X / 2
    #
    #             left_noise = max(0, int(noise_index * self.DELTA_X - self.WAVE_LENGTH / 2))
    #             right_noise = min(orig_trace.shape[1] ,int(noise_index * self.DELTA_X + self.WAVE_LENGTH / 2))
    #             if left_noise >= right_noise:
    #                 continue
    #
    #             left_current = max(0, int(j * self.DELTA_X - self.WAVE_LENGTH / 2))
    #             right_current = min(orig_trace.shape[1], int(j * self.DELTA_X + self.WAVE_LENGTH / 2))
    #             if left_current >= right_current:
    #                 continue
    #
    #             noise_data = orig_trace[:, left_noise : right_noise]
    #             current_data = orig_trace[:, left_current : right_current]
    #
    #             noise_data_ampl = filtered_trace[:, left_noise: right_noise]
    #             current_data_ampl = filtered_trace[:, left_current: right_current]
    #
    #             amplitude_noise = self.__RMS3(noise_data_ampl)
    #             amplitude_current = self.__RMS3(current_data_ampl)
    #             relative_amplitude = amplitude_current / amplitude_noise
    #
    #             low_noise = self.__RMS3(scipy.signal.sosfilt(sos_low_filter, noise_data))
    #             high_noise = self.__RMS3(scipy.signal.sosfilt(sos_high_filter, noise_data))
    #             energy_noise = low_noise / (high_noise + self.EPS)
    #
    #             low_current = self.__RMS3(scipy.signal.sosfilt(sos_low_filter, current_data))
    #             high_current = self.__RMS3(scipy.signal.sosfilt(sos_high_filter, current_data))
    #             energy_current = low_current / (high_current + self.EPS)
    #
    #             relative_energy = energy_current / energy_noise
    #
    #             # multiplying_amplitude = 1
    #             # if relative_amplitude >= 1.5:
    #             #     multiplying_amplitude = 1.2
    #             # elif relative_amplitude >= 3:
    #             #     multiplying_amplitude = 1.5
    #
    #             multiplying_amplitude = 1 + 0.5 / (1 + np.exp(-2 * (relative_amplitude - 1.5)))
    #
    #             # multiplying_energy = 1
    #             # if relative_energy >= 2:
    #             #     multiplying_energy = 1.2
    #
    #             multiplying_energy = 1 + 0.2 / (1 + np.exp(-2 * (relative_energy - 2)))
    #
    #             coeff = multiplying_amplitude * multiplying_energy
    #             print(f"mul ampl {multiplying_amplitude} \n"
    #                   f"mul en {multiplying_energy} \n"
    #                   f"for i - {i} ({index * self.DELTA_X}) and j - {j}")
    #
    #             prediction[j, 1] *= coeff
    #
    #     return prediction
    #
    # def __RMS3(self, window: npt.NDArray[npt.Shape["3, *"], npt.Float32]) -> float:
    #     rms_value_n = np.sqrt(np.mean(np.square(window[0])))
    #     rms_value_e = np.sqrt(np.mean(np.square(window[1])))
    #     rms_value_z = np.sqrt(np.mean(np.square(window[2])))
    #     return (rms_value_n + rms_value_e + rms_value_z) / 3.0

    def __horizontal_2D_sliding_window(self, array: npt.NDArray[npt.Shape["3, *"], npt.Float32],
                                       sliding_window_size: tuple[int, int], dx: int = 40) -> npt.NDArray[
        npt.Shape["*, 3, 400"], npt.Float32
    ]:
        shape: tuple[int, int, int] = ((array.shape[-1] - sliding_window_size[-1]) // dx + 1,) + sliding_window_size
        strides: tuple[int, int, int] = (array.strides[-1] * dx,) + array.strides[-2:]
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides, writeable=False)


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, progress_bar: QProgressBar, overall_size: int) -> None:
        keras.callbacks.Callback.__init__(self)
        self.progress_bar: QProgressBar = progress_bar
        temp_count: int = int(overall_size / NeuralNetworkModel.BATCH_SIZE)
        self.count_size: int = temp_count if temp_count > 0 else 1

    def on_predict_begin(self, logs=None) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def on_predict_end(self, logs=None) -> None:
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

    def on_predict_batch_end(self, batch, logs=None) -> None:
        current_percentage: int = int(batch / self.count_size * 100)
        self.progress_bar.setValue(current_percentage)
