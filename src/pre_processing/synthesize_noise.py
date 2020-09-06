from scipy import signal
import numpy as np

from src.utils import read_config


# parameters ---------------------------------------------
config_ini = read_config.read_config()
amplitude = config_ini.getint('NOISE', 'amplitude_h')
high_h = config_ini.getint('BANDPASS', 'high_h')
high_l = config_ini.getint('BANDPASS', 'high_l')
side = config_ini.getint('BANDPASS', 'side_w')
b_gstop = config_ini.getint('BANDPASS', 'gstop')
b_gpass = config_ini.getint('BANDPASS', 'gpass')
fs = config_ini.getint("SAMPLING_FREQUENCY", 'fs')
# --------------------------------------------------------


def band_pass(data):
    gpass = b_gpass
    gstop = b_gstop
    Wp1 = high_l / (fs / 2)
    Wp2 = high_h / (fs / 2)
    Ws1 = (high_l + side) / (fs / 2)
    Ws2 = (high_h + side) / (fs / 2)
    N1, Wn1 = signal.buttord([Wp1, Wp2], [Ws1, Ws2], gpass, gstop)
    b1, a1 = signal.butter(N1, Wn1, "bandpass")
    return signal.filtfilt(b1, a1, data)


def run(data):
    """
    :param data: # data = ndarray, (a single signal)
    :return: add noise signal
    """
    length = np.linspace(0, len(data))
    noise = amplitude * np.random.rand(len(length))
    noise = band_pass(noise)
    return data + noise