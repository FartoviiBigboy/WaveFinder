import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import regularizers
from kapre import STFT, Magnitude, MagnitudeToDecibel

from Seismogram import Seismogram


class MaxABSScaler(keras.layers.Layer):
    """
    Rescale to [-1,1]
    """

    def __init__(self):
        super(MaxABSScaler, self).__init__()

    def call(self, inputs):
        min_abs_val = tf.abs(tf.reduce_min(inputs))
        max_abs_val = tf.abs(tf.reduce_max(inputs))
        max_abs = tf.maximum(min_abs_val, max_abs_val)
        return inputs / max_abs


class NeuralNetworkModel:
    WAVE_LENGTH = 400
    NUMBER_OF_TRACES = 3
    DELTA_X = 20
    BATCH_SIZE = 32

    def __init__(self):
        self.device_for_calculation = "/GPU:0" if (len(tf.config.list_physical_devices('GPU')) > 0) else "/device:CPU:0"
        print(self.device_for_calculation)
        self.model = self.__initialize_model()
        self.__load_model_weights()

    def __load_model_weights(self):
        self.model.load_weights("resources/mymodel_3_15.h5")

    def get_prediction(self, seismogram: Seismogram, progress_bar):

        traces_copy = np.array(
            [
                np.concatenate((np.zeros(200), seismogram.traces[0], np.zeros(200))),
                np.concatenate((np.zeros(200), seismogram.traces[1], np.zeros(200))),
                np.concatenate((np.zeros(200), seismogram.traces[2], np.zeros(200))),
            ]
        )
        converted_traces = self.__horizontal_2D_sliding_window(
            traces_copy,
            (self.NUMBER_OF_TRACES, self.WAVE_LENGTH),
            self.DELTA_X)
        converted_traces = np.transpose(converted_traces, (0, 2, 1))
        callbacks = CustomCallback(progress_bar, converted_traces.shape[0])

        predicted = self.model.predict(converted_traces[:], batch_size=NeuralNetworkModel.BATCH_SIZE,
                                       callbacks=[callbacks], verbose=0)
        p_der_indexes = self.__get_maximums(predicted[:, 0])
        s_der_indexes = self.__get_maximums(predicted[:, 1])
        return predicted, p_der_indexes, s_der_indexes

    def __initialize_model(self):
        with tf.device(self.device_for_calculation):
            output_size = 3
            input_size = (400, 3)

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

            model = tf.keras.models.Model(inputs=input_layer, outputs=x, name='custom_Resnet34')

            fLoss = keras.losses.SparseCategoricalCrossentropy()
            fOptimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            fMetric = [keras.metrics.SparseCategoricalAccuracy()]

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

    def __get_maximums(self, array):
        return [i
                for i in range(1, len(array) - 1)
                if (array[i] - array[i - 1] > 0 >= array[i + 1] - array[i])]

    def __horizontal_2D_sliding_window(self, array, sliding_window_size, dx=40):
        shape = array.shape[:-2] + ((array.shape[-1] - sliding_window_size[-1]) // dx + 1,) + sliding_window_size
        strides = array.strides[:-2] + (array.strides[-1] * dx,) + array.strides[-2:]
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides, writeable=False)


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, progress_bar, overall_size):
        keras.callbacks.Callback.__init__(self)
        self.progress_bar = progress_bar
        temp_count = int(overall_size / NeuralNetworkModel.BATCH_SIZE)
        self.count_size = temp_count if temp_count > 0 else 1

    def on_predict_begin(self, logs=None):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def on_predict_end(self, logs=None):
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

    def on_predict_batch_end(self, batch, logs=None):
        current_percentage = int(batch / self.count_size * 100)
        self.progress_bar.setValue(current_percentage)
