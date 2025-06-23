#!/usr/bin/python

"""
Simple test script for Indiana Drones project.
Allows testing specific parts or test cases.
"""

import sys
import unittest
from testing_suite_indiana_drones import PartATestCase, PartBTestCase, IndianaDronesTestResult

def test_part_a_only():
    """Run only Part A tests."""
    print("Running Part A tests only...")
    suite = unittest.TestSuite()
    suite.addTest(PartATestCase('test_case1'))
    suite.addTest(PartATestCase('test_case2'))
    suite.addTest(PartATestCase('test_case3'))
    suite.addTest(PartATestCase('test_case4'))
    suite.addTest(PartATestCase('test_case5'))
    
    result = IndianaDronesTestResult(stream=sys.stdout)
    suite.run(result)
    print(f"Part A Score: {result.avg_credit * 100:.2f}%")

def test_part_b_only():
    """Run only Part B tests."""
    print("Running Part B tests only...")
    suite = unittest.TestSuite()
    suite.addTest(PartBTestCase('test_case1'))
    suite.addTest(PartBTestCase('test_case2'))
    suite.addTest(PartBTestCase('test_case3'))
    suite.addTest(PartBTestCase('test_case4'))
    suite.addTest(PartBTestCase('test_case5'))
    
    result = IndianaDronesTestResult(stream=sys.stdout)
    suite.run(result)
    print(f"Part B Score: {result.avg_credit * 100:.2f}%")

def test_specific_case(part, case_num):
    """Run a specific test case."""
    if part.upper() == 'A':
        test_class = PartATestCase
        part_name = "Part A"
    elif part.upper() == 'B':
        test_class = PartBTestCase
        part_name = "Part B"
    else:
        print("Invalid part. Use 'A' or 'B'")
        return
    
    print(f"Running {part_name} Test Case {case_num}...")
    suite = unittest.TestSuite()
    suite.addTest(test_class(f'test_case{case_num}'))
    
    result = IndianaDronesTestResult(stream=sys.stdout)
    suite.run(result)
    print(f"{part_name} Test Case {case_num} Score: {result.avg_credit * 100:.2f}%")

def test_with_verbose():
    """Run tests with verbose output."""
    print("Running tests with verbose output...")
    # Temporarily enable verbose mode
    import testing_suite_indiana_drones
    original_verbose = testing_suite_indiana_drones.VERBOSE_FLAG
    testing_suite_indiana_drones.VERBOSE_FLAG = True
    
    try:
        test_part_a_only()
        print("\n" + "="*50 + "\n")
        test_part_b_only()
    finally:
        testing_suite_indiana_drones.VERBOSE_FLAG = original_verbose

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_specific.py part_a          # Test Part A only")
        print("  python test_specific.py part_b          # Test Part B only")
        print("  python test_specific.py case A 1        # Test Part A, Case 1")
        print("  python test_specific.py case B 3        # Test Part B, Case 3")
        print("  python test_specific.py verbose         # Test with verbose output")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "part_a":
        test_part_a_only()
    elif command == "part_b":
        test_part_b_only()
    elif command == "case" and len(sys.argv) >= 4:
        part = sys.argv[2]
        case_num = int(sys.argv[3])
        test_specific_case(part, case_num)
    elif command == "verbose":
        test_with_verbose()
    else:
        print("Invalid command. Use 'part_a', 'part_b', 'case', or 'verbose'")
        sys.exit(1) 