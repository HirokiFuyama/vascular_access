import numpy as np
import glob
import soundfile as sf
from utils import util_func
import configparser


def read_config():
    config_ini = configparser.ConfigParser()
    config_ini.read("config.ini")
    return config_ini


def read_audio(directory_path):
    """
    :param directory_path:
    :return:List of array of raw data.
    """
    file_path = glob.glob(directory_path)
    file_path.sort()
    raw_list = [sf.read(file_path[i])[0] for i in range(len(file_path))]
    return raw_list


def cut_overlap(raw_list):
    """
    This function cut data to 30 seconds and overlap data to increase it.
    :param raw_list: list of array of raw data.
    :return: type: array, shape: (batch, fs*window_length)
    """
    # parameters ------------------------------------------------------------------
    config_ini = read_config()
    window_length = config_ini.getint('PRE_PROCESSING', 'window_length')  # sec
    slide_length = config_ini.getint('PRE_PROCESSING', 'slide_length')  # sec
    cut_length = config_ini.getint('PRE_PROCESSING', 'cut_length')  # sec (30/2)
    fs = config_ini.getint('PRE_PROCESSING', 'fs')  # Hz
    # ------------------------------------------------------------------------------
    cut_list = [raw_list[i][int(len(raw_list[i])/2 - fs*cut_length): int(len(raw_list[i])/2 + fs*cut_length)] for i in range(len(raw_list))]
    overlapped = []
    for sig in cut_list:
        sig_over = [sig[i: i + int(fs*window_length)] for i in range(0, len(sig)-int(window_length*fs), int(slide_length*fs))]
        overlapped.append(sig_over)
    overlapped = np.array(overlapped)
    return overlapped.reshape(int(overlapped.shape[0]*overlapped.shape[1]), overlapped.shape[2])


def stft_spectrogram(data):
    """
    This function is short time fourier transform with hamming window.
    :param data
    :return:
    """
    # parameters -------------------------------------------------------------
    config_ini = read_config()
    window_length = config_ini.getint('STFT', 'window_length')
    slide_length = config_ini.getint('STFT', 'slide_length')
    high_freq =
    low_freq =
    bins =
    # ------------------------------------------------------------------------
    power_list = []
    for i in range(0, int(len(data)-window_length*fs), (int(slide_length*fs))):
        power, freq = util_func.power_spectrum_gain(util_func.hamming_window(data[i: i+int(fs*window_length)]), fs)
        index_low = int(low_freq/freq[1])
        index_high = int(high_freq/freq[1])
        power = power[index_low:index_high]
        power_mean = [np.array(power[j:j+bins]).mean() for j in range(0, len(power)-bins, bins)]
        power_list.append(power_mean)
    return util_func.min_max_norm(power_list)


# test-----------------------------------------------------------
dire_path = '/Users/hiroki/github/vascular_access/data/A/*.wav'
fs = 441000


sig = read_audio(dire_path)
print(len(sig))
import matplotlib.pyplot as plt
p,f = util_func.power_spectrum_gain(sig[0],fs)
plt.plot(f,p)
plt.show()

