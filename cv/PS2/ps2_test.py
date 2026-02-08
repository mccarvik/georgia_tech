import os
import unittest

import cv2
import numpy as np

import ps2

INPUT_DIR = "tests/"


class DFT(unittest.TestCase):
    def test_dft(self):
        x = np.load(INPUT_DIR + "dft_input.npy")
        check_result = np.load(INPUT_DIR + "dft_output.npy")

        ps_result = ps2.dft(x)

        self.assertTrue(np.allclose(ps_result, check_result))

    def test_idft(self):
        x = np.load(INPUT_DIR + "idft_input.npy")
        check_result = np.load(INPUT_DIR + "idft_output.npy")

        ps_result = ps2.idft(x)

        self.assertTrue(np.allclose(ps_result, check_result))

    # @unittest.skip("demonstrating skipping")
    def test_dft2(self):
        x = np.load(INPUT_DIR + "dft2_input.npy")
        check_result = np.load(INPUT_DIR + "dft2_output.npy")

        ps_result = ps2.dft2(x)

        self.assertTrue(np.allclose(ps_result, check_result))

    def test_idft2(self):
        x = np.load(INPUT_DIR + "idft2_input.npy")
        check_result = np.load(INPUT_DIR + "idft2_output.npy")

        ps_result = ps2.idft2(x)

        self.assertTrue(np.allclose(ps_result, check_result))

    def test_compression(self):
        x = np.load(INPUT_DIR + "compression_input.npy")

        ps_result, _ = ps2.compress_image_fft(x, 0.5)

        check_result = np.load(INPUT_DIR + "compression_output.npy")
        self.assertTrue(np.allclose(ps_result, check_result))

    def test_low_pass_filter(self):
        x = np.load(INPUT_DIR + "lpf_input.npy")

        ps_result, _ = ps2.low_pass_filter(x, 100)

        check_result = np.load(INPUT_DIR + "lpf_output.npy")
        self.assertTrue(np.allclose(ps_result, check_result))


if __name__ == "__main__":
    unittest.main()
