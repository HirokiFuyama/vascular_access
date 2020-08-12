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
    :return:List of array of files path.
    """
    file_path = glob.glob(directory_path)
    file_path.sort()
    raw_list = [sf.read(file_path[i])[0] for i in range(len(file_path))]
    return raw_list


def cut_out_data(df_raw, window_length, slide_length, fs):
    window_length = 10  # sec
    slide_length = 1  # sec
    #------------------------------------------------------------------------------
    signal_30sec = [df_raw[i][round(round(len(df_raw[i]) / 2) - fs * 15): round(round(len(df_raw[i]) / 2) + fs * 15)] for i in range(len(df_raw))]
    signal_cut = []
    for sig in signal_30sec:
        sig_cut = [sig[i:i+fs*window_length] for i in range(0,len(sig)-window_length*fs,int(slide_length*fs))]
        signal_cut.append(sig_cut)
    signal_cut = np.array(signal_cut)
    signal_cut_slide = signal_cut.reshape(int(signal_cut.shape[0]*signal_cut.shape[1]), signal_cut.shape[2])

def stft_spectrogram(data, window_length, slide_length, low_freq, high_freq, bins, fs):
    """
    This function is short time fourier transform with hamming window.
    :param data
    :param window_length: unit second
    :param slide_length: unit second
    :param low_freq: unit Hz (low of frequency range for spectrogram)
    :param high_freq: unit Hz (high of frequency range for spectrogram)
    :param bins : int (bin of frequency)
    :param fs : sampling frequency
    :return:
    """
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

