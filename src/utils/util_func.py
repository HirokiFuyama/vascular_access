from scipy import fftpack
from scipy import signal
import numpy as np


def power_spectrum_gain(data, fs):
    """
    :param data:
    :param fs: unit Hz
    :return: power spectrum, frequency (fs/2)
    """
    n = len(data)
    y = fftpack.fft(data) / n
    y = y[0:round(n / 2)]
    power = 2 * (np.abs(y) ** 2)
    power = 10 * np.log10(power)
    freq = np.arange(0, fs / 2, fs / (n-1))
    return power, freq


def hamming_window(data):
    win_hamming = signal.hamming(len(data))
    return data * win_hamming


def min_max_image(data):
    """
    :param data: shape:(batch,)
    :return: Normalized data , shape:same size as input, range 0 to 255
    """
    data = np.array(data)
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)
    return (data-x_min)/(x_max-x_min)*(255-0)
