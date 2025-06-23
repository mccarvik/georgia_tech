#!/usr/bin/python

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

import unittest
import random
import math
import time
import traceback
import hashlib
import copy
import string
from typing import List, Dict
import sys

from test_cases import IndianaDronesPartATestCases, IndianaDronesPartBTestCases

try:
    import indiana_drones
    studentExc = None
except Exception as e:
    studentExc = traceback.format_exc()
import drone

PI = math.pi

########################################################################
# for debugging set the time limit to a big number
########################################################################
TIME_LIMIT = 10  # seconds

########################################################################
# set to True for lots-o-output, also passed into drone under test
########################################################################
VERBOSE_FLAG = False

########################################################################
# set to True to disable multiprocessing while running in a debugger and
# to ensure there is no stochasticity in grading. Set to False by default.
########################################################################
DEBUGGING_SINGLE_PROCESS = False

########################################################################
# TODO: you can set NOISE_FLAG to false during development
# but be sure to run and test with noise = True
# before submitting your solution.
########################################################################
NOISE_FLAG = True
NOISE_MOVE = 0.01

if DEBUGGING_SINGLE_PROCESS:
    import multiprocessing.dummy as mproc
else:
    import multiprocessing as mproc

########################################################################
# used to generate unique ids for landmarks.  will change for grader
########################################################################
HASH_SEED = 'some_seed'
#random.seed(HASH_SEED)

PART_A_CREDIT = 0.60
PART_B_CREDIT = 0.40

# DO NOT REMOVE THESE VARIABLES.
PART_A_SCORE = None
PART_B_SCORE = None


class Submission:
    """Student Submission.

    Attributes:
        submission_action_plan(Queue): Student score of executed action plan.
        submission_error(Queue): Error messages generated during executed action plan.
        submission_reported_tree_locations(Queue): log of tree locations reported by the extract action used for grading.
    """

    def __init__(self):

        # if DEBUGGING_SINGLE_PROCESS:
        #     import queue
        #     self.time_left = queue.Queue(1)
        #     self.submission_action_plan = queue.Queue(1)
        #     self.submission_error = queue.Queue(1)
        #     self.submission_reported_tree_locations = queue.Queue(1)
        # else:
        self.time_left = mproc.Manager().Queue(1)
        self.submission_action_plan = mproc.Manager().Queue(1)
        self.submission_error = mproc.Manager().Queue(1)
        self.submission_reported_tree_locations = mproc.Manager().Queue(1)

    def _reset(self):
        """Reset submission results.
        """
        while not self.time_left.empty():
            self.submission_action_plan.get()
        self.time_left.put(True)

        while not self.submission_action_plan.empty():
            self.submission_action_plan.get()

        while not self.submission_error.empty():
            self.submission_error.get()

        while not self.submission_reported_tree_locations.empty():
            self.submission_reported_tree_locations.get()

    def execute_student_plan(self, area_map: List[list], tree_radius: List[list], max_distance: float = 1.0,
                             max_steering: float = PI / 2. + 0.01, drone_distance_noise: float = 0.05,
                             drone_bearing_noise: float = 0.02, horizon: float = float('inf')):
        """Execute student plan and store results in submission.

        Args:
            area_map: the area map to test against.
            tree_radius : the radii of the trees on the map
            max_distance: maximum distance per move.
            max_steering: maximum steering per move.
            drone_distance_noise: distance noise to set for Drone.
            drone_bearing_noise: bearing noise to set for Drone.
            horizon: distance of max measurement
        """
        self._reset()

        state = State(area_map,
                      tree_radius,
                      max_distance,
                      max_steering,
                      measure_distance_noise=drone_distance_noise,
                      measure_bearing_noise=drone_bearing_noise,
                      horizon=horizon)

        if VERBOSE_FLAG:
            print('Initial State:')
            print(state)

        try:
            student_planner = indiana_drones.IndianaDronesPlanner(max_distance, max_steering)

            state_output = ''
            time_left = self.time_left.get()

            while len(state.collected_treasure) < 1 and time_left:
                state_output += str(state)
               
                ret = student_planner.next_move(state.generate_measurements(), state._treasure_loc_fromstart)
               
                if isinstance(ret, str):
                    action = ret
                else:
                    action, locs = ret

                state.update_according_to(action)
                if VERBOSE_FLAG:
                    print(state)
                    print('Time Left:')
                    print(time_left)

                if not self.time_left.empty():
                    time_left = self.time_left.get()

            if VERBOSE_FLAG:
                print('Final State:')
                print(state)

            self.submission_action_plan.put((state.collected_treasure, state.crashes))
            if not time_left:
                error_message = ("Time limit reached: You didn't complete the task quick enough." +
                f'It was expected to finish in fewer than {TIME_LIMIT} second(s).')
                self.submission_error.put(error_message)

        except Exception:
            self.submission_error.put(traceback.format_exc())
            self.submission_action_plan.put(([],[]))
            
            


class State:
    """Current State.

    Args:
        area_map: the area map.
        tree_radius: the list of tree radii
        max_distance:  the max distance the drone can travel in a single move.
        max_steering:  the max steering angle the drone can turn in a single move.
        measure_distance_noise: Noise of the distance measurement
        measure_bearing_noise: Noise of the bearing measurement
        horizon: distance of max measurement

    Attributes:
        collected_treasure:  treasure successfully extracted.
        max_distance:   max distance the drone can travel in one move.
        max_steering:   the max steering angle the drone can turn in a single move.
        _start_position: location of initial drone placement
    """
    EXTRACTION_DISTANCE = 0.25
    WAIT_PENALTY = 0.1  # seconds

    def __init__(self, area_map: List[list], tree_radius: List[list], max_distance: float = 1.0,
                 max_steering: float = PI / 2. + 0.01, measure_distance_noise: float = 0.05,
                 measure_bearing_noise: float = 0.03, horizon: float = float('inf')):

        self.collected_treasure = []
        self.crashes = []
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.tree_locs_on_map = []
        self.horizon = horizon

        rows = len(area_map)
        cols = len(area_map[0])

        self._start_position = dict()
        self._treasure_location = dict()
        self._treasure_loc_fromstart = dict() 

        # Now process the interior of the provided map
        for i in range(rows):
            for j in range(cols):
                this_square = area_map[i][j]
                x, y = float(j), -float(i)

                # Process trees
                if this_square in string.ascii_uppercase:
                    tree = {'id': int(hashlib.md5((str(this_square) + str(random.random())).encode('utf-8')).hexdigest(),
                                     16),
                           'x': x + 0.5,
                           'y': y - 0.5,
                           'type': this_square,
                           'radius': tree_radius[this_square]}

                    self.tree_locs_on_map.append(tree)

                # Process start
                elif this_square == '@':
                    self._start_position['x'] = x + 0.5
                    self._start_position['y'] = y - 0.5

                elif this_square == '*':
                    self._treasure_location['x'] = x + 0.5
                    self._treasure_location['y'] = y - 0.5
                    self._treasure_location['type'] = this_square
    
        treasure_x_fromstart = self._treasure_location['x'] - self._start_position['x']
        treasure_y_fromstart = self._treasure_location['y'] - self._start_position['y']
        
        
        self._treasure_loc_fromstart['type'] = self._treasure_location['type']
        self._treasure_loc_fromstart['x'] = treasure_x_fromstart
        self._treasure_loc_fromstart['y'] = treasure_y_fromstart


        # initialize the drone at the start position and at a bearing pointing due east
        self.drone = drone.Drone(x=self._start_position['x'],
                                 y=self._start_position['y'],
                                 bearing=0.0,
                                 max_distance=self.max_distance,
                                 max_steering=self.max_steering,
                                 measure_distance_noise=measure_distance_noise,
                                 measure_bearing_noise=measure_bearing_noise)

    def generate_measurements(self, noise: bool = NOISE_FLAG):
        """Generate measurements of trees on map.

        Args:
            noise: Move with noise if True.
                Default: NOISE_FLAG

        Returns:
            Measurements to trees in the format:
                {'unique tree id':{'distance': 0.0, 'bearing': 0.0, 'type': 'A', 'radius':0.5}, ...}
        """
        measurements = dict()

        # process trees
        for location in self.tree_locs_on_map:
            distance, bearing = self.drone.measure_distance_and_bearing_to((location['x'], location['y']), noise=noise)
            if distance - location['radius'] < self.horizon:
                measurements[location['id']] = {'distance': distance,
                                                'bearing': bearing,
                                                'type': location['type'],
                                                'radius': location['radius']}
        return measurements

    def line_circle_intersect(self, first_point, second_point, origin, radius):
        """ Checks if a line segment between two points intersects a circle of a certain radius and origin

        Args: 
            first_point : (x,y)
            second_point : (x,y)
            origin : (x,y)
            radius : r

        Returns:
            intersect : True/False
        
        """
        
        ###REFERENCE###
        #https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        x1, y1 = first_point
        x2, y2 = second_point
        
        ox,oy = origin
        r = radius
        x1 -= ox
        y1 -= oy
        x2 -= ox
        y2 -= oy
        a = (x2 - x1)**2 + (y2 - y1)**2
        b = 2*(x1*(x2 - x1) + y1*(y2 - y1))
        c = x1**2 + y1**2 - r**2
        disc = b**2 - 4*a*c

        if a == 0:
            if c <= 0:
                return True
            else:
                return False
        else: 

            if (disc <= 0):
                return False
            sqrtdisc = math.sqrt(disc)
            t1 = (-b + sqrtdisc)/(2*a)
            t2 = (-b - sqrtdisc)/(2*a)
            if((0 < t1 and t1 < 1) or (0 < t2 and t2 < 1)):
                return True
            return False

    def check_crash(self, old_position, new_position):
        """ Checks if a line segment between old and new position intersects with any of the trees on the map and updates list of crashes

        Args: 
            old_position : (x,y)
            new_position : (x,y)

        """

        for tree in self.tree_locs_on_map:            
                if tree['type'] not in self.crashes:
                    origin = (tree['x'], tree['y'])
                    if self.line_circle_intersect(old_position, new_position, origin, tree['radius']):
                        self.crashes.append(tree['type'])

    
    def update_according_to(self, action: str, noise: bool = NOISE_FLAG):
        """Update state according to action.

        Args:
            action: action to execute.
            noise: Move with noise if True.
                Default: NOISE_FLAG

        Raises:
            Exception: if improperly formatted action.
        """
        action = action.split()
        action_type = action[0]

        if action_type == 'move':
            distance, steering = action[1:]
            old_position = (self.drone.x, self.drone.y)
            self._attempt_move(float(distance), float(steering), noise=noise)
            new_position = (self.drone.x, self.drone.y)
            self.check_crash(old_position, new_position)

        elif action_type == 'extract':
            try:
                treasure_type = action[1]
                estimate_x = float(action[2])
                estimate_y = float(action[3])
                self._attempt_extraction(treasure_type, estimate_x, estimate_y)
            except IndexError:
                # improper move format: kill test
                raise Exception(f"improperly formatted action: {' '.join(action)}")

        else:
            # improper move format: kill test
            raise Exception(f"improperly formatted action: {' '.join(action)}")

    def _attempt_move(self, distance: float, steering: float, noise: bool = NOISE_FLAG):
        """Attempt move action if valid.

        The drone may move between 0 and max_distance
        The drone may turn between -max_steering and +max_steering

        Illegal moves - the drone will not move
        - Moving a distance outside of [0,max_distance]
        - Steering angle outside [-max_steering, max_steering]

        Args:
            steering: Angle to turn before moving.
            distance: Distance to travel.

        Raises:
            ValueError: if improperly formatted move destination.
        """
        try:
            distance_ok = 0.0 <= distance <= self.max_distance
            steering_ok = (-self.max_steering) <= steering <= self.max_steering

            if noise:
                steering += random.gauss(0.0, NOISE_MOVE)
                distance += random.gauss(0.0, NOISE_MOVE)

            if distance_ok and steering_ok:
                self.drone.move(distance, steering, True)
            else:
                print(f'The command is ignored because it is outside of the acceptable distance and/or steering range!')

        except ValueError:
            raise Exception(f"improperly formatted move command : {distance} {steering}")

    def _attempt_extraction(self, treasure_type: str, estimate_x: float, estimate_y: float):
        """Attempt to extract treasure from the current x,y location.

        Extract treasure if current location is within EXTRACTION_DISTANCE of specified treasure_type.
        Otherwise, pause for WAIT_PENALTY
        """
  
        if treasure_type == '*':
            drone_distance = drone.compute_distance((self.drone.x, self.drone.y),
                                                    (self._treasure_location['x'], self._treasure_location['y']))

            translated_x = estimate_x + self._start_position['x']
            translated_y = estimate_y + self._start_position['y']

            estimate_distance = drone.compute_distance((translated_x, translated_y),
                                                        (self._treasure_location['x'], self._treasure_location['y']))

            if drone_distance <= self.EXTRACTION_DISTANCE and estimate_distance <= self.EXTRACTION_DISTANCE:
                self.collected_treasure.append(self._treasure_location)
                return

        time.sleep(self.WAIT_PENALTY)

        if VERBOSE_FLAG:
            print(f"*** Location ({self.drone.x}, {self.drone.y}) does not contain the treasure type <{treasure_type}> within "
                  f"the extraction distance.")

    def __repr__(self):
        """Output state object as string.
        """
        output = '\n'
        output += 'Drone State:\n'
        output += f'\t x = {self.drone.x:6.2f}, y = {self.drone.y:6.2f}, hdg = {self.drone.bearing * 180. / PI:6.2f}\n'
        if len(self.collected_treasure) > 0 :
            collected = self.collected_treasure[0]
            collected = collected['type']
            output += f'Treasure Extracted: {collected}\n'
        else:
            output += f'Treasure Extracted: None\n'
        output += f'Crashes Detected: {self.crashes}\n'

        return output


class IndianaDronesTestResult(unittest.TestResult):

    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super(IndianaDronesTestResult, self).__init__(stream, verbosity, descriptions)
        self.stream = stream
        self.credit = []
        self.results = []

    def stopTest(self, test):
        super(IndianaDronesTestResult, self).stopTest(test)
        try:
            self.credit.append(test.last_credit)
            self.results.append(test.last_result)
            self.stream.write(test.last_result + '\n')

        except AttributeError as exp:
            self.stream.write(str(exp))

    @property
    def avg_credit(self):
        try:
            return sum(self.credit) / len(self.credit)

        except Exception:
            return 0.0


class PartATestCase(unittest.TestCase):
    """Test PartA
    """
    results_file = 'results_partA.txt'

    results = ['', 'PART A TEST CASE RESULTS']
    SCORE_TEMPLATE = "\n".join((
        "\nPart A Test Case {test_case} Results",
        "  Expected Location:\t{expected}",
        "  SLAM Location:\t{location}",
        "  Credit: {score:.0%}"
        "\n\n- - - - END OF TEST CASE - - - -\n",
    ))
    FAIL_TEMPLATE = "\n".join((
        "\nPart A Test Case {test_case} Results",
        "  Failed: {message}",
        "  Expected Location:\t{expected}",
        "  SLAM Location:\t{location}",
        "  Credit: {score:.0%}"
        "\n\n- - - - END OF TEST CASE - - - -\n",
    ))

    credit = []

    def setUp(self):

        self.last_result = ''
        self.last_credit = 0.0

        if studentExc:
            self.last_result = str(studentExc)
            raise studentExc

    def run_with_params(self, params: Dict):
        """Run test case using desired parameters.
        Args:
            params: a dictionary of test parameters.
        """

        state = State(params['area_map'],
                      params['tree_radius'],
                      measure_distance_noise=params['drone_distance_noise'],
                      measure_bearing_noise=params['drone_bearing_noise'],
                      horizon=params['horizon'])
        drone_dist_error = float('inf')
        landmark_dist_errors = dict()

        state_beliefs = list()
        ground_truth = list()

        try:
            indiana_drones_slam = indiana_drones.SLAM()

            # calculate drone position error
            for move in params['move']:
                meas = state.generate_measurements()
                indiana_drones_slam.process_measurements(meas)

                action = move.split()
                state.update_according_to(move)
                indiana_drones_slam.process_movement(float(action[1]), float(action[2]))

                coordinates = indiana_drones_slam.get_coordinates()
                belief = coordinates['self']
                truth = (state.drone.x - state._start_position['x'],
                         state.drone.y - state._start_position['y'])

                drone_dist_error = drone.compute_distance(belief, truth)

                if VERBOSE_FLAG:
                    print("Current Belief:", belief)
                    print("True Position:", truth)
                    print("Error:", drone_dist_error, "\n")

                state_beliefs.append(belief)
                ground_truth.append(truth)

            # calculate landmark errors
            for landmark in state.tree_locs_on_map:
                coordinates = indiana_drones_slam.get_coordinates()
                student_landmark_x, student_landmark_y = coordinates[landmark['id']]

                translated_x = student_landmark_x + state._start_position['x']
                translated_y = student_landmark_y + state._start_position['y']

                landmark_dist_errors[landmark['id']] = {'error': drone.compute_distance((translated_x, translated_y),
                                                                              (landmark['x'], landmark['y'])), 
                                                        'type': landmark['type']}

        except Exception as exp:
            
            self.last_result = self.FAIL_TEMPLATE.format(message=traceback.format_exc(),
                                                         expected="exception",
                                                         location="exception",
                                                         score=0.0,
                                                         **params)
            self.last_credit = 0.0
            self.fail(str(exp))

        max_drone_score = 0.5
        max_landmark_score = 0.5

        drone_score = 0.0
        landmark_score = 0.0

        # calculate score for drone distance error
        if drone_dist_error < params['drone_tolerance']:
            drone_score += max_drone_score

        # calculate score for landmark distance errors
        missed_landmarks = list()
        for landmark_type, landmark_error in landmark_dist_errors.items():
            if landmark_error['error'] < params['landmark_tolerance']:
                landmark_score += max_landmark_score / len(state.tree_locs_on_map)
            else:
                missed_landmarks.append({'landmark': landmark_error['type'],
                                         'error': landmark_error['error']})

        drone_score = round(drone_score, 5)
        landmark_score = round(landmark_score, 5)

        total_score = drone_score + landmark_score

        if total_score >= 1.0:
            result = self.SCORE_TEMPLATE.format(expected=ground_truth,
                                                location=state_beliefs,
                                                score=total_score, **params)
        else:
            if drone_score < max_drone_score:
                drone_message = f"Drone location error {drone_dist_error} is greater than {params['drone_tolerance']}. "
            else:
                drone_message = ''

            if landmark_score < max_landmark_score:
                landmark_message = f"Landmark location errors are greater than {params['landmark_tolerance']}\n{missed_landmarks}"
            else:
                landmark_message = ''

            result = self.FAIL_TEMPLATE.format(message=drone_message + landmark_message,
                                               expected=ground_truth,
                                               location=state_beliefs, score=total_score, **params)

        self.last_result = result
        self.last_credit = total_score

        self.assertTrue(drone_score >= max_drone_score,
                        f"Drone location error {drone_dist_error} is greater than {params['drone_tolerance']}")

        self.assertTrue(landmark_score >= max_landmark_score,
                        f"Landmark location errors are greater than {params['landmark_tolerance']}\n{missed_landmarks}")

    def test_case1(self):
        self.run_with_params(IndianaDronesPartATestCases.test_case_1)
    def test_case2(self):
        self.run_with_params(IndianaDronesPartATestCases.test_case_2)
    def test_case3(self):
        self.run_with_params(IndianaDronesPartATestCases.test_case_3)
    def test_case4(self):
        self.run_with_params(IndianaDronesPartATestCases.test_case_4)
    def test_case5(self):
        self.run_with_params(IndianaDronesPartATestCases.test_case_5)

class PartBTestCase(unittest.TestCase):
    """ Test PartB.
    """
    results_file = 'results_partB.txt'

    results = ['', 'PART B TEST CASE RESULTS']
    SCORE_TEMPLATE = "\n".join((
        "\nPart B Test Case {test_case} Results",
        "  Needed treasure: {needed_treasure}",
        "  Collected treasure:{collected}",
        "  Tree Crashes: {crashes}",
        "  Credit: {score:.0%}"
        "\n\n- - - - END OF TEST CASE - - - -\n",
    ))
    FAIL_TEMPLATE = "\n".join((
        "\nPart B Test Case {test_case} Results",
        "  Failed: {message}",
        "  Needed treasure: {needed_treasure}",
        "  Tree Crashes: {crashes}",
        "  Collected treasure:{collected}",
        "  Credit: {score:.0%}"
        "\n\n- - - - END OF TEST CASE - - - -\n",
    ))

    credit = []

    def setUp(self):
        """Initialize test setup.
        """
        self.last_result = ''
        self.last_credit = 0.0
        if studentExc:
            self.last_result = str(studentExc)
            raise studentExc
        self.student_submission = Submission()

    def check_results(self, params: Dict, error_message: str):

        #extracted_treasure = {}
        collected_treasure = []
        collected = False

        extracted_treasure_type = ''
        extraction_score = 0
        crash_score = 0
        # Get number of treasure collected
        if not self.student_submission.submission_action_plan.empty():
            #extracted_treasure, crashes = self.student_submission.submission_action_plan.get()
            collected_treasure, crashes = self.student_submission.submission_action_plan.get()

        #if len(extracted_treasure) > 0:
        #    extracted_treasure_type = extracted_treasure['type']
        #    if extracted_treasure_type == '*':
        #        extraction_score = 0.5
        if len(collected_treasure) > 0:
            collected = True
            extraction_score = 0.5

        crash_score = max(0, 0.5-(0.25* len(crashes)))

        if extraction_score > 0:
            score = extraction_score + crash_score
        else:
            score = 0

        if not self.student_submission.submission_error.empty():
            error_message = self.student_submission.submission_error.get()
            #result = self.FAIL_TEMPLATE.format(message=error_message, needed_treasure = ['*'], collected=extracted_treasure_type, crashes=crashes, score=score, **params)
            result = self.FAIL_TEMPLATE.format(message=error_message, needed_treasure = ['*'], collected=collected, crashes=crashes, score=score, **params)
        else:
            #result = self.SCORE_TEMPLATE.format(needed_treasure = ['*'], collected=extracted_treasure_type, crashes=crashes, score=score, **params)
            result = self.SCORE_TEMPLATE.format(needed_treasure = ['*'], collected=collected, crashes=crashes, score=score, **params)

        #return result, score, error_message, extracted_treasure_type, crashes
        return result, score, error_message, collected, crashes

    def run_with_params(self, params: Dict):
        """Run test case using desired parameters.
        Args:
            params: a dictionary of test parameters.
        """
        sys.stdout.write(f'~ ~ ~ Start of test case # {params["test_case"]} ~ ~ ~\n\n')

        error_message = ''

        if DEBUGGING_SINGLE_PROCESS:
            try:
                self.student_submission.execute_student_plan(params['area_map'],
                                                             params['tree_radius'],
                                                             params['max_distance'],
                                                             params['max_steering'],
                                                             params['drone_distance_noise'],
                                                             params['drone_bearing_noise'],
                                                             params['horizon'])
            except Exception as exp:
                error_message = exp

            #result, score, error_message, extracted_treasure_needed, crashes = self.check_results(params, error_message)
            result, score, error_message, collected, crashes = self.check_results(params, error_message)

        else:
            test_process = mproc.Process(target=self.student_submission.execute_student_plan,
                                         args=(params['area_map'],
                                               params['tree_radius'],
                                               params['max_distance'],
                                               params['max_steering'],
                                               params['drone_distance_noise'],
                                               params['drone_bearing_noise'],
                                               params['horizon']))

            try:
                test_process.start()
                test_process.join(TIME_LIMIT)
                self.student_submission.time_left.put(False)     # notify child process to finish
                test_process.join(1)                             # give child process a second to wrap things up
            
            except Exception as exp:
                error_message = exp
                
                

            # If test still running then terminate
            if test_process.is_alive():
                test_process.terminate()
                error_message = ('Test ended unexpectedly! No extracted treasure data available')
                result = self.FAIL_TEMPLATE.format(message=error_message, needed_treasure = ['*'], collected=False, crashes = [], score=0.0, **params)
                score = 0.0
                extracted_treasure_needed = '*'
            else:
                result, score, error_message, collected, crashes = self.check_results(params, error_message)
        
        self.last_result = result
        self.last_credit = score

        self.assertFalse(error_message, error_message)
        self.assertTrue(round(score, 7) == 1.0,
                        f"The treasure was extracted and all obstacles were avoided") 

    def test_case1(self):
        self.run_with_params(IndianaDronesPartBTestCases.test_case_1)
    def test_case2(self):
        self.run_with_params(IndianaDronesPartBTestCases.test_case_2)
    def test_case3(self):
        self.run_with_params(IndianaDronesPartBTestCases.test_case_3)
    def test_case4(self):
        self.run_with_params(IndianaDronesPartBTestCases.test_case_4)
    def test_case5(self):
        self.run_with_params(IndianaDronesPartBTestCases.test_case_5)


def run_all(stream):
    suites = map(lambda case: unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(case)),
                 [PartATestCase, PartATestCase, PartATestCase, PartATestCase, PartATestCase, PartATestCase, PartATestCase, PartATestCase, PartATestCase, PartATestCase,
                  PartBTestCase, PartBTestCase, PartBTestCase, PartBTestCase, PartBTestCase, PartBTestCase, PartBTestCase, PartBTestCase, PartBTestCase, PartBTestCase])

    avgs = []
    for suite in suites:
        result = IndianaDronesTestResult(stream=stream)
        suite.run(result)
        avgs.append(result.avg_credit)

    partA = avgs[0:10]
    partB = avgs[10:20]

    #remove lowest score and take average
    partA.remove(min(partA))
    partB.remove(min(partB))

    results = [sum(partA)/9.0, sum(partB)/9.0]
    stream.write('part A score: %.02f\n' % (results[0] * 100))
    stream.write('part B score: %.02f\n' % (results[1] * 100))

    weights = (PART_A_CREDIT, PART_B_CREDIT)
    total_score = round(sum(results[i] * weights[i] for i in (0, 1)) * 100)
    stream.write('score: %.02f\n' % total_score)


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


# Only run all of the test automatically if this file was executed from the command line.
# Otherwise, let Nose/py.test do it's own thing with the test cases.
if __name__ == "__main__":
    if studentExc:
        print(studentExc)
        print('score: 0')
    else:
        student_id = indiana_drones.who_am_i()
        if student_id:
            run_all(sys.stdout)
        else:
            print("Student ID not specified.  Please fill in 'whoami' variable.")
            print('score: 0')
