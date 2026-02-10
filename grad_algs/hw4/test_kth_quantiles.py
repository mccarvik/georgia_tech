import unittest

from cs6515_kth_quantiles import KthQuantiles


class TestKthQuantiles(unittest.TestCase):
    def test_base_case_1(self):
        quantiles = KthQuantiles(
            [1, 2, 3, 4],
            2,
        )

        self.assertEqual(quantiles, [2])

    def test_base_case_2(self):
        quantiles = KthQuantiles(
            [1, 2, 3, 4, 5, 6, 7, 8],
            4,
        )

        self.assertEqual(quantiles, [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
