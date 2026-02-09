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
        diff = np.abs(ps_result - check_result)
        max_diff = np.max(diff)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print("\n=== Compression Debugging ===")
        print(f"Max abs diff: {max_diff}")
        print(f"Location of max diff: {max_idx}")
        print(f"ps_result at max_idx: {ps_result[max_idx]}")
        print(f"check_result at max_idx: {check_result[max_idx]}")
        print(f"Allclose? {np.allclose(ps_result, check_result)}")
        print(f"Total number of mismatched (>1e-5): {(diff > 1e-5).sum()}")

        # Detailed debugging: show how many cells in each channel are over the threshold
        threshold = 1e-5
        if diff.ndim == 3:
            num_channels = diff.shape[2]
            print("Cells over threshold per channel:")
            for ch in range(num_channels):
                over = (diff[:,:,ch] > threshold).sum()
                print(f"  Channel {ch}: {over} cells over threshold")
        elif diff.ndim == 2:
            over = (diff > threshold).sum()
            print(f"2D array: {over} cells over threshold")
        else:
            over = (diff > threshold).sum()
            print(f"1D array: {over} cells over threshold")

        # Added per-instruction debugging: show individual cells with largest differences
        # Let's display the top 5 mismatches, sorted descendingly by absolute diff
        num_to_show = 5
        flat_indices = np.argsort(diff.ravel())[::-1][:num_to_show]
        if flat_indices.size > 0:
            print(f"Top {num_to_show} cells with largest absolute differences:")
            for i, flat_idx in enumerate(flat_indices, 1):
                idx = np.unravel_index(flat_idx, diff.shape)
                print(
                    f"[{i}] At index {idx}: "
                    f"ps_result={ps_result[idx]}, "
                    f"check_result={check_result[idx]}, "
                    f"abs diff={diff[idx]}"
                )

        self.assertTrue(np.allclose(ps_result, check_result))

    def test_low_pass_filter(self):
        x = np.load(INPUT_DIR + "lpf_input.npy")

        ps_result, _ = ps2.low_pass_filter(x, 100)

        check_result = np.load(INPUT_DIR + "lpf_output.npy")
        self.assertTrue(np.allclose(ps_result, check_result))


if __name__ == "__main__":
    unittest.main()
