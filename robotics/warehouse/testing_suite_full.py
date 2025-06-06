######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

import io
import sys
import unittest
import traceback

try:
    from testing_suite_partA import PartATestCase
    from testing_suite_partB import PartBTestCase
    from testing_suite_partC import PartCTestCase
    from warehouse import who_am_i
    studentExc = None
except Exception as e:
    studentExc = traceback.format_exc()


def run_all( fout ):

    partA_weight = 40 # 40% of total
    partB_weight = 40 # 40% of total
    partC_weight = 20 # 20% of total
    totalScore = 0.0

    scores = []
    weights = [partA_weight, partB_weight, partC_weight]
    parts = ['A', 'B', 'C']

    for part_name, test_cases, weight in zip(parts, [PartATestCase, PartBTestCase, PartCTestCase], weights):

        fout.write(f'{"-_" * 20}\n    part {part_name} starting    \n{"-_" * 20}\n')

        part_fout = io.StringIO()
        test_cases.fout = part_fout

        suite = unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(test_cases))

        # get the number of test cases in this test-suite
        no_of_test_cases = suite.countTestCases()

        # get the test result
        result = unittest.TestResult()
        suite.run(result)

        # printout for individual test case details
        output = part_fout.getvalue()

        # loop through the credit for each test case and scale it to the points per test case, this will help the overall score for this part
        # such that if some test cases may have received more than max credit they can help other test cases that may have received less credit
        score = (sum(test_cases.credit) / no_of_test_cases) * weight

        # calculate total score for this part, and cap it at the weight for this part
        # we need the cap so one test-suite does not dominate the overall score
        score = min(weight, score)

        # append the score for later printing
        scores.append(score)

        fout.write( output )
        fout.write( f'part {part_name} result: {score:.02f}% [weight = {int(weight)}%]\n')
        fout.write( f'{"-_"*20}\n    part {part_name} finished    \n{"-_"*20}\n')

    fout.write('-----------------------------\n')
    for part, score, weight in zip(parts, scores, weights):
        fout.write(f'Part {part} result: {score:.02f}% [weight = {weight}%]\n')

    totalScore = sum(scores)
    intTotalScore = int( (totalScore) + .5)   # round score up to nearest integer
    if intTotalScore > 100:
        intTotalScore = 100
    fout.write( "score: %d\n" % intTotalScore )


# This flag is used to check whether project files listed in the json have been modified.
# Modifications include (but are not limited to) print statements, changing flag values, etc.
# If you have modified the project files in some way, the results may not be accurate.
# Turn file_checker on by setting the flag to True to ensure you are running against
# the same framework as the Gradescope autograder.
file_checker = False  # set to True to turn file checking on

if file_checker:
    import json
    import hashlib
    import pathlib
    print("File checking is turned on.")
    with open('file_check.json', 'r') as openfile:
        json_dict = json.load(openfile)

    modified_files = []
    for file in json_dict:
        f = str(file)
        try:
            current = pathlib.Path(file).read_text().replace(' ', '').replace('\n', '')
            file_hash = hashlib.sha256(current.encode()).hexdigest()
            if file_hash != json_dict[f]:
                modified_files.append(f)
        except:
            print(f'File ({f}) not in project folder.')

    if len(modified_files) == 0:
        print("You are running against the same framework as the Gradescope autograder.")
    else:
        print("Warning. The following files have been modified and the results may not be accurate:")
        print(", ".join(modified_files))

if __name__ == '__main__':
    if studentExc:
        print(studentExc)
        print('score: 0')
    else:
        student_id = who_am_i()
        if student_id:
            try:
                run_all( sys.stdout )
            except Exception as e:
                print(e)
                print('score: 0')
        else:
            print("Student ID not specified.  Please fill in 'whoami' variable.")
            print('score: 0')
