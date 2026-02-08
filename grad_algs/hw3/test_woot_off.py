import unittest

from cs6515_woot_off import WootOff


class TestWootOff(unittest.TestCase):
    def test_base_case_1(self):
        weights = (1, 2, 3)
        price = (4, 9, 6)

        capacity = (2, 3, 5)

        total, profits = WootOff(weights, price, capacity)

        self.assertEqual(total, 44)

        self.assertListEqual(profits, [9, 13, 22])


if __name__ == "__main__":
    unittest.main()
