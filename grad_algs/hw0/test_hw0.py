import unittest

from cs6515_hw0 import HW0


class TestHW0(unittest.TestCase):
    def test_base_case_1(self):
        size, result = HW0((1,))

        self.assertEqual(size, 1)
        self.assertEqual(result, [0])

    def test_base_case_2(self):
        size, result = HW0((1, 2, 3, 4))

        self.assertEqual(size, 4)
        self.assertEqual(result, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
