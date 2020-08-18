import numpy as np
import glob
import sys
import soundfile as sf
from PIL import Image

from utils import util_func
from utils import read_config


def read_audio():
    """
    :return:List of array of raw data.
    """
    # parameters ------------------------------------------------
    config_ini = read_config.read_config()
    directory_path = config_ini.get('PATH', 'data_directory')
    # -----------------------------------------------------------
    file_path = glob.glob(directory_path)
    file_path.sort()
    raw_data = [sf.read(file_path[i])[0] for i in range(len(file_path))]
    if file_path == []:
        print('FileNotFoundError: No such file or directory: ', file=sys.stderr)
        sys.exit(1)
    return raw_data


def cut_overlap(raw_data):
    """
    This function cut data to 30 seconds and overlap data to increase it.
    :param raw_data: list of array of raw data.
    :return: type: array, shape: (batch, fs*window_length)
    """
    # parameters ------------------------------------------------------------------
    config_ini = read_config.read_config()
    window_length = config_ini.getint('PRE_PROCESSING', 'window_length')  # sec
    slide_length = config_ini.getint('PRE_PROCESSING', 'slide_length')  # sec
    cut_length = config_ini.getint('PRE_PROCESSING', 'cut_length')  # sec (30/2)
    fs = config_ini.getint('SAMPLING_FREQUENCY', 'fs')  # Hz
    # ------------------------------------------------------------------------------
    cut_list = [raw_data[i][int(len(raw_data[i]) / 2 - fs * cut_length): int(len(raw_data[i]) / 2 + fs * cut_length)] for i in range(len(raw_data))]
    overlapped = []
    for sig in cut_list:
        sig_over = [sig[i: i + int(fs*window_length)] for i in range(0, len(sig)-int(window_length*fs), int(slide_length*fs))]
        overlapped.append(sig_over)
    overlapped = np.array(overlapped)
    return overlapped.reshape(int(overlapped.shape[0]*overlapped.shape[1]), overlapped.shape[2])


def stft_spectrogram(data):
    """
    This function calculate the spectrogram using STFT with hamming window.
    :param data
    :return:
    """
    # parameters -------------------------------------------------------------
    config_ini = read_config.read_config()
    window_length = config_ini.getint('STFT', 'window_length')
    slide_length = config_ini.getfloat('STFT', 'slide_length')
    high_freq = config_ini.getint('STFT', 'frequency_high')
    low_freq = config_ini.getint('STFT', 'frequency_low')
    bins = config_ini.getint('STFT', 'bins')
    fs = config_ini.getint('SAMPLING_FREQUENCY', 'fs')
    # ------------------------------------------------------------------------
    power_list = []
    for i in range(0, int(len(data)-window_length*fs), (int(slide_length*fs))):
        power, freq = util_func.power_spectrum(util_func.hamming_window(data[i: i+int(fs*window_length)]), fs)
        index_low = int(low_freq/freq[1])
        index_high = int(high_freq/freq[1])
        power = power[index_low:index_high]
        power_mean = [np.array(power[j:j+bins]).mean() for j in range(0, len(power)-bins, bins)]
        power_list.append(power_mean)
    return util_func.min_max_image(power_list)


def padding(spectrogram_array):
    """
    :param spectrogram_array: shape==(,,,)
    :return:
    """
    # parameters -------------------------------------
    cofig_ini = read_config.read_config()
    image_len = cofig_ini.getint("IMAGE", 'shape')
    # ------------------------------------------------
    diff_row = spectrogram_array.shape[1] - image_len
    diff_col = spectrogram_array.shape[2] - image_len
    try:
        if diff_row <= 0 and diff_col <= 0:
            spectrogram = np.pad(spectrogram_array, [(0, 0), (0, abs(diff_row)), (0, abs(diff_col))], 'constant')

        elif diff_row <= 0 and diff_col >= 0:
            spectrogram = np.pad(spectrogram_array, [(0, 0), (0, abs(diff_row)+diff_col), (0, 0)], 'constant')

        elif diff_row >= 0 and diff_col <= 0:
            spectrogram = np.pad(spectrogram_array, [(0, 0), (0, 0), (0, abs(diff_col)+diff_row)], 'constant')

        elif diff_row >= 0 and diff_col >= 0:
            diff = diff_row - diff_col
            if diff <= 0:
                spectrogram = np.pad(spectrogram_array, [(0, 0), (0, abs(diff)), (0, 0)], 'constant')
            elif diff >= 0:
                spectrogram = np.pad(spectrogram_array, [(0, 0), (0, 0), (0, diff)], 'constant')
    except:
        print("Error: Can't padding :", file=sys.stderr)
        sys.exit(1)
    return spectrogram


def spectrogram_image(data):
    """
    padding a spectrogram to square and saving it as an image.
    :param data: return of stft_spectrogram function.
    :return: save image.
    """
    # parameters ------------------------------------------
    config_ini = read_config.read_config()
    save_path = config_ini.get('PATH', 'save_image')
    # -----------------------------------------------------
    spectrogram = [stft_spectrogram(i) for i in data]
    spectrogram = np.array(spectrogram)
    spectrogram = padding(spectrogram)
    for i in range(len(spectrogram)):
        spe_array = spectrogram[i].T
        image = Image.fromarray(spe_array.astype(np.uint8))
        image.save(save_path + "spectrogram{}.png".format(i))
        # For test, cheack image ----------------------------------------------------
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()
        # if i == 5:
        #     break


def run():
    raw_data = read_audio()
    cut_data = cut_overlap(raw_data)
    return spectrogram_image(cut_data)


# test-----------------------------------------------------------
# run()