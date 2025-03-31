import numpy as np
import obspy
import pywt
import matplotlib.pyplot as plt

from Seismogram import Seismogram


def apply_cwt(signal, sampling_rate=100, wavelet='cmor', scales=np.arange(1, 64)):
    """
    Apply Continuous Wavelet Transform (CWT) to a signal.

    Parameters:
        signal (array): Input seismic signal.
        sampling_rate (float): Sampling rate of the signal (Hz).
        wavelet (str): Mother wavelet to use (e.g., 'cmor' for complex Morlet).
        scales (array): Scales to use for the wavelet transform.

    Returns:
        cwt_coeffs (array): Wavelet coefficients (time-scale representation).
        frequencies (array): Corresponding frequencies for each scale.
    """
    # Compute CWT
    cwt_coeffs, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1 / sampling_rate)

    return cwt_coeffs, frequencies


def extract_energy_in_band(cwt_coeffs, frequencies, freq_range=(1, 5)):
    """
    Extract energy in a specific frequency band from CWT coefficients.

    Parameters:
        cwt_coeffs (array): Wavelet coefficients.
        frequencies (array): Frequencies corresponding to scales.
        freq_range (tuple): Frequency range (min, max) to extract energy.

    Returns:
        energy (array): Energy in the specified frequency band over time.
    """
    # Find indices corresponding to the frequency range
    freq_indices = np.where((frequencies >= freq_range[0]) & (frequencies <= freq_range[1]))[0]

    # Sum energy across the selected frequency range
    energy = np.sum(np.abs(cwt_coeffs[freq_indices, :]) ** 2, axis=0)

    return energy


# Example usage
sampling_rate = 100  # Hz
st = obspy.read("C:\\Users\\Nick\\Desktop\\уч 8 сем\\ВКР нейронка\\my\\check\\records\\3.mseed")
sorted_list = sorted(st, key=lambda x: (x.stats.station, x.stats.channel))
sorted_list[::3], sorted_list[1::3] = sorted_list[1::3], sorted_list[::3]

seis = Seismogram(sorted_list[0:0 + 3], "C:\\Users\\Nick\\Desktop\\уч 8 сем\\ВКР нейронка\\my\\check\\records\\3.mseed")
seis.apply_filter(2, 0.5, "high")

seismic_signal = seis.traces[0]  # Replace with actual seismic data
scales = np.arange(1, 64)  # Define scales (frequency resolution)

cwt_coeffs, frequencies = apply_cwt(seismic_signal, sampling_rate, scales=scales)

energy_1_to_5_hz = extract_energy_in_band(cwt_coeffs, frequencies, freq_range=(1, 5))

# Plot the energy over time
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(seismic_signal)) / sampling_rate, energy_1_to_5_hz, label='1–5 Hz Energy')
plt.title('Energy in 1–5 Hz Band')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend()
plt.show()
