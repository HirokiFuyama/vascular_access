import unittest
from pre_processing import convert_spectrogram


class TestConvertSpectrogram(unittest.TestCase):

    def test_convert_spectrogram(self):
        """
        :return:
        """
        test_file_path = '/vascular_access/data/test/*.wav'
        expected = [105, 105, 111, 119, 132, 144, 145, 158, 159, 149, 136, 143, 142,
                    144, 143, 138, 130, 118, 102, 108, 117, 121, 120, 110, 119, 129,
                    148, 165, 174, 176, 173, 167, 162, 156, 146, 144, 127, 128, 140,
                    153, 163, 172, 175, 171, 164, 155, 147, 145, 147, 151, 161, 166,
                    168, 165, 168, 168, 168, 167, 164, 159, 155, 150, 143, 122, 152,
                    159, 159, 163, 166, 168, 169, 165, 156, 149, 161, 163, 160, 158,
                    164, 158, 142, 143, 147, 154, 158, 156, 159, 162, 160, 152, 146,
                    143, 138, 134, 150, 162, 173, 177, 175, 171, 170, 168, 164, 154,
                    150, 157, 158, 151, 140, 130, 128, 139, 143, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        actual = convert_spectrogram.unit_test(test_file_path)
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
