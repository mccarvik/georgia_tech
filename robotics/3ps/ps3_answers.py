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

######################################################################
# How to use:
#
#   For fill-in questions, change the value of each answer variable.
#
#   For programming questions, change the implementation of the
#   function provided.
#
#   Run the file from the command line to check:
#
#       python ps3_answers.py
#
######################################################################

from math import *
import random
import checkutil

# NOTE: All questions are presented in the problem set videos (lesson modules), please watch these to get the full
# context on the questions below.  Following each question video is a subsequent video with the answer.

# If you see different scores locally and on Gradescope this may be an indication
# that you are uploading a different file than the one you are executing locally.
# If this local ID doesn't match the ID on Gradescope then you uploaded a different file.
OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib
    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')

seed = 0  # This seed results in successful results for python's rng
random.seed(seed)

# STUDENT ID

# Please specify your GT login ID in the whoami variable (ex: jsmith225).

whoami = 'kmccarville3'

# QUESTION 1: EMPTY CELL

# What is the probability that zero particles are in state A?

q1_n_1  = 0.75
q1_n_4  = 0.316    # rounded to 3 decimal places
q1_n_10 = 0.0563   # rounded to 4 decimal places

# QUESTION 2: MOTION QUESTION
#
# Given: There are 5 particles in A, 3 in B, 3 in C, and 1 in D.
# We now take a motion step. Particles move with 50% chance horizontally, and
# 50% chance vertically. They never move diagonally or stay in the same cell.
#
# How many particles do we expect in cell A, B, C, and D after 1 step?
# After infinite steps?
#
q2_1_step_A = 3
q2_1_step_B = 3
q2_1_step_C = 3
q2_1_step_D = 3
q2_infinite_step_A = 3
q2_infinite_step_B = 3
q2_infinite_step_C = 3
q2_infinite_step_D = 3

# QUESTION 3: SINGLE PARTICLE
#
# What will happen if we run a particle filter with N=1 particle?
# Put 0 for unchecked, 1 for checked
#
q3_works_fine                 = 0
q3_ignores_robot_measurements = 1
q3_ignores_robot_motion       = 0
q3_it_likely_fails           = 1
q3_none_of_above             = 0

# QUESTION 4: CIRCULAR MOTION

def q4_move(self, motion):
    """
    This function is the move function for the robot class.
    It takes a motion vector and returns a new robot object with the correct coordinates.
    """
    # You can replace the INSIDE of this function with the move function you modified in the module quiz
    # return res # make sure your move function returns an instance
    # of the robot class with the correct coordinates.
    
    # motion[0] = steering angle
    # motion[1] = distance
    
    # Create a new robot object
    print(self.length)
    print(self.bearing_noise)
    res = robot(self.length)
    # get all the giveen values
    res.set_noise(self.bearing_noise, self.steering_noise, self.distance_noise)
    
    # Extract motion parameters
    # get the steering angle and distance from the motion vector
    steering = motion[0]
    distance = motion[1]
    
    # add the noises
    # Apply steering noise
    steering2 = random.gauss(steering, self.steering_noise)
    # Apply distance noise
    distance2 = random.gauss(distance, self.distance_noise)
    
    # Calculate turning radius
    # If steering is 0, the robot moves straight
    # Otherwise, it moves in a circle with radius = length/tan(steering)
    if abs(steering2) < 0.001:
        # Straight motion
        res.x = self.x + (distance2 * cos(self.orientation))
        res.y = self.y + (distance2 * sin(self.orientation))
        res.orientation = self.orientation
    else:
        # now we just calculate the turning radius cuz we know we are not going straight
        # Circular motion
        radius = self.length / tan(steering2)
        beta = distance2 / radius
        
        # Calculate center of the circle
        # standard trig shiz here
        cx = self.x - (sin(self.orientation) * radius)
        cy = self.y + (cos(self.orientation) * radius)
        # Update position
        res.x = cx + (sin(self.orientation + beta) * radius)
        res.y = cy - (cos(self.orientation + beta) * radius)
        res.orientation = (self.orientation + beta) % (2.0 * pi)
    return res

# QUESTION 5: SENSING

def q5_sense(self, add_noise=1):
    # You can ignore add_noise for Q5
    landmarks = globals()['landmarks']  # local variable to reference landmarks data
    Z = []

    for i in range(len(landmarks)):
        # took the hint on atan2 function
        bearing = atan2(landmarks[i][0] - self.y, landmarks[i][1] - self.x) - self.orientation
        # ignoring noise here
        bearing %= 2.0 * pi
        Z.append(bearing)
    
    return Z

# QUESTION 6: FINAL QUIZ

def q6_move(self, motion):
    """
    This function is the move function for the robot class.
    It takes a motion vector and returns a new robot object with the correct coordinates.
    """
    # copied from above
    # Create a new robot object
    res = robot(self.length)
    res.set_noise(self.bearing_noise, self.steering_noise, self.distance_noise)
    
    # Extract motion parameters
    steering = motion[0]
    distance = motion[1]
    
    # adds noise
    # Apply steering noise
    steering2 = random.gauss(steering, self.steering_noise)
    # Apply distance noise
    distance2 = random.gauss(distance, self.distance_noise)
    
    # same as above
    # Calculate turning radius
    if abs(steering2) < 0.001:
        # Straight motion
        res.x = self.x + (distance2 * cos(self.orientation))
        res.y = self.y + (distance2 * sin(self.orientation))
        res.orientation = self.orientation
    else:
        # Circular motion
        radius = self.length / tan(steering2)
        beta = distance2 / radius
        
        # Calculate center of the circle
        cx = self.x - (sin(self.orientation) * radius)
        cy = self.y + (cos(self.orientation) * radius)
        
        # Update position
        res.x = cx + (sin(self.orientation + beta) * radius)
        res.y = cy - (cos(self.orientation + beta) * radius)
        res.orientation = (self.orientation + beta) % (2.0 * pi)
    return res


# Most of this came from the videos
# I just added the noise to the sense function


def q6_sense(self, add_noise=1):
    """
    This function is the sense function for the robot class.
    It takes a boolean add_noise and returns a list of bearings to the landmarks.
    """
    landmarks = globals()['landmarks']
    Z = []
    
    for landmark in landmarks:
        # Calculate the bearing to each landmark
        dx = landmark[1] - self.x
        dy = landmark[0] - self.y
        
        # Calculate bearing relative to robot's orientation
        bearing = atan2(dy, dx) - self.orientation
        
        # Normalize bearing to [-pi, pi]
        bearing = (bearing + pi) % (2.0 * pi) - pi
        
        # Add noise if requested
        # same as above but now we are adding noise
        if add_noise:
            bearing = random.gauss(bearing, self.bearing_noise)
            # Normalize again after adding noise
            bearing = (bearing + pi) % (2.0 * pi) - pi
        
        Z.append(bearing)
    
    return Z

######################################################################
# Grading methods
#
# Do not modify code below this point.
#
# The auto-grader does not use any of the below code
#
######################################################################
landmarks  = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]] # landmarks are in (y, x) format
world_size = 100.0
tolerance_xy = 15.0 # Tolerance for localization in the x and y directions.
tolerance_orientation = 0.25 # Tolerance for orientation.
class robot(object):
    move_func_name = None   # used to control which move function is used by robot instance
    sense_func_name = None  # used to control which sense function is used by robot instance

    def __init__(self, length=10.0):
        self.x = random.random() * world_size # initial x position
        self.y = random.random() * world_size # initial y position
        self.orientation = random.random() * 2.0 * pi # initial orientation
        self.length = length # length of robot
        self.bearing_noise  = 0.0 # initialize bearing noise to zero
        self.steering_noise = 0.0 # initialize steering noise to zero
        self.distance_noise = 0.0 # initialize distance noise to zero
        self.move_func = globals().get(robot.move_func_name)
        self.sense_func = globals().get(robot.sense_func_name)

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))

    def set(self, new_x, new_y, new_orientation):

        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):
        self.bearing_noise  = float(new_b_noise)
        self.steering_noise = float(new_s_noise)
        self.distance_noise = float(new_d_noise)

    def measurement_prob(self, measurements):
        predicted_measurements = self.sense(0)

        if not isinstance(predicted_measurements, list) and not isinstance(predicted_measurements, tuple):
            raise RuntimeError( 'sense() expected to return a list, instead got %s' % type(predicted_measurements) )

        if len(predicted_measurements) != len(measurements):
            raise RuntimeError( '%d measurements but %d predicted measurements from sense'
                                % (len(measurements), len(predicted_measurements)) )

        error = 1.0
        for i in range(len(measurements)):
            error_bearing = abs(measurements[i] - predicted_measurements[i])
            error_bearing = (error_bearing + pi) % (2.0 * pi) - pi

            error *= ((exp((- (error_bearing ** 2) / (self.bearing_noise ** 2)) / 2.0) /
                      sqrt(2.0 * pi * (self.bearing_noise ** 2))))

        return error

    def move(self, motion):
        if self.move_func is not None:
            return self.move_func(self, motion)
        else:
            return self

    def sense(self, add_noise=1):
        if self.sense_func is not None:
            return self.sense_func(self, add_noise)
        else:
            return []

def get_position(p):
    x = 0.0
    y = 0.0
    orientation = 0.0
    for i in range(len(p)):
        x += p[i].x
        y += p[i].y
        orientation += (((p[i].orientation - p[0].orientation + pi) % (2.0 * pi))
                        + p[0].orientation - pi)
    return [(x / len(p)), (y / len(p)), (orientation / len(p))]




FILL_IN_TEST_CASES = ({'variable_name': 'q1_n_1',
                       'str_func':          checkutil.float_to_str,
                       'answer_hash':       '42f2d990b0d1b069fcc6952dcf8e03f5',
                       'points_avail':      1./3.},
                      {'variable_name': 'q1_n_4',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       'cd6b6a8e6aca6a3bf5b914b2eb1d471e',
                       'points_avail':      1./3.},
                      {'variable_name': 'q1_n_10',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '162d0da51e5b91c22b8ef816a33b93ae',
                       'points_avail':      1./3.},
                      {'variable_name': 'q2_1_step_A',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '4c519e55122fb9103f3429d4d35246c2',
                       'points_avail':      1./8.},
                      {'variable_name': 'q2_1_step_B',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '865f680967fee7bab877431d55a667e9',
                       'points_avail':      1./8.},
                      {'variable_name': 'q2_1_step_C',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '48482b0046ab77768ab1471544d39c03',
                       'points_avail':      1./8.},
                      {'variable_name': 'q2_1_step_D',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       'b67dd96cc694c9beac33d9fbd459174c',
                       'points_avail':      1./8.},
                      {'variable_name': 'q2_infinite_step_A',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       'eff0c2c81ea4c663110ccca7da9783c9',
                       'points_avail':      1./8.},
                      {'variable_name': 'q2_infinite_step_B',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '3e74f80dcde13f3e2d4016ccfdb3e6fd',
                       'points_avail':      1./8.},
                      {'variable_name': 'q2_infinite_step_C',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '32e9394cabcb64eb6fe926fd18f7eea1',
                       'points_avail':      1./8.},
                      {'variable_name': 'q2_infinite_step_D',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       'b6a07a19e607d0686dfc627c139c373f',
                       'points_avail':      1./8.},
                      {'variable_name': 'q3_works_fine',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       'ab5378bc6072dac87d8eae185c1d5a50',
                       'points_avail':      1./5.},
                      {'variable_name': 'q3_ignores_robot_measurements',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '85f15f9ddfad3555ce7f803375591fcb',
                       'points_avail':      1./5.},
                      {'variable_name': 'q3_ignores_robot_motion',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       'a16592a0343fcca9b90b5220ac53491b',
                       'points_avail':      1./5.},
                      {'variable_name': 'q3_it_likely_fails',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       '945f27265e22c511bb56f1631610442a',
                       'points_avail':      1./5.},
                      {'variable_name': 'q3_none_of_above',
                          'str_func':          checkutil.float_to_str,
                       'answer_hash':       'c885bdcdfbd4a9f353ee68f7be7dbf9e',
                       'points_avail':      1./5.}, )

# PROGRAMMING

def grader_q4(length=20,
              x = 0.0,
              y = 0.0,
              orientation = 0.0,
              bearing_noise = 0.0,
              steering_noise = 0.0,
              distance_noise = 0.0,
              motions = ()):
    robot.move_func_name = 'q4_move'

    outputs = []

    grader_robot = robot(length)
    grader_robot.set(x, y, orientation)
    grader_robot.set_noise(bearing_noise, steering_noise, distance_noise)

    for motion in motions:
        grader_robot = grader_robot.move(motion)
        outputs.append((grader_robot.x, grader_robot.y, grader_robot.orientation))

    return outputs

def grader_q5(length=20,
              x = 0.0,
              y = 0.0,
              orientation = 0.0,
              bearing_noise = 0.0,
              steering_noise = 0.0,
              distance_noise = 0.0):

    robot.sense_func_name = 'q5_sense'

    grader_robot = robot(length)
    grader_robot.set(x, y, orientation)
    grader_robot.set_noise(bearing_noise, steering_noise, distance_noise)
    return grader_robot.sense()


def grader_q6(length=20,
              bearing_noise = 0.0,
              steering_noise = 0.0,
              distance_noise = 0.0,
              motions=(),
              measurements=(),
              N=100):

    robot.move_func_name = 'q6_move'
    robot.sense_func_name = 'q6_sense'
    result = ''
    points_earned = 0

    p = []

    for i in range(N):
        r = robot(length=length)
        r.set_noise(bearing_noise, steering_noise, distance_noise)
        p.append(r)

    for t in range(len(motions)):
        p2 = []
        for i in range(N):
            p2.append(p[i].move(motions[t]))
        p = p2

        w = []
        for i in range(N):
            w.append(p[i].measurement_prob(measurements[t]))

        p3 = []
        index = int(random.random() * N) % N
        beta = 0.0
        mw = max(w)
        for i in range(N):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N
            p3.append(p[index])
        p = p3

    return get_position(p)

def q4_output_match(values, expected):
    temp = True
    if len(values) != len(expected):
        return False
    else:
        for t in range(len(values)):
            if not (abs(values[t][0] - expected[t][0]) < 0.01 and abs(values[t][1] - expected[t][1]) < 0.01 and abs(values[t][2] - expected[t][2]) < 0.01):
                temp = False
                break
    return temp

def q5_output_match(values, expected):
    temp = True
    if len(values) != len(expected):
        return False
    else:
        for t in range(len(values)):
            if not (abs(values[t] - expected[t]) < 0.01):
                temp = False
                break
    return temp

def q6_check_output(pos0, pos1):

    error_x = abs(pos0[0] - pos1[0])
    error_y = abs(pos0[1] - pos1[1])
    error_orientation = abs(pos0[2] - pos1[2])
    error_orientation = (error_orientation + pi) % (2.0 * pi) - pi
    correct = error_x < tolerance_xy and error_y < tolerance_xy \
              and error_orientation < tolerance_orientation
    return correct

CODE_TEST_CASES = ({'function_name': 'grader_q4',
                    'function_input': dict(length=20.,
                                           x=0.0,
                                           y=0.0,
                                           orientation=0.0,
                                           bearing_noise=0.0,
                                           steering_noise=0.0,
                                           distance_noise=0.0,
                                           motions=[(0.0, 10.0), (pi / 6.0, 10), (0.0, 20.0)]),
                    'expected_output': [(10.0, 0.0, 0.0), (19.861, 1.4333, 0.2886), (39.034, 7.127, 0.2886)],
                    'outputs_match_func': lambda actual, expected: q4_output_match(actual, expected),
                    'output_to_str_func': checkutil.list_of_float_to_str},
                    {'function_name': 'grader_q5',
                    'function_input': dict(length=20.,
                                           x=30.0,
                                           y=20.0,
                                           orientation=0.0,
                                           bearing_noise=0.0,
                                           steering_noise=0.0,
                                           distance_noise=0.0),
                    'expected_output': [6.004885648174475, 3.7295952571373605, 1.9295669970654687, 0.8519663271732721],
                    'outputs_match_func': lambda actual, expected: q5_output_match(actual, expected),
                    'output_to_str_func': checkutil.list_of_float_to_str},
                    {'function_name': 'grader_q6',
                    'function_input': dict(length=20.,
                                           bearing_noise=0.1,
                                           steering_noise=0.1,
                                           distance_noise=5.0,
                                           motions=[((2. * pi / 10), 20.) for row in range(8)],
                                           measurements=[
                                               [4.746936, 3.859782, 3.045217, 2.045506],
                                               [3.510067, 2.916300, 2.146394, 1.598332],
                                               [2.972469, 2.407489, 1.588474, 1.611094],
                                               [1.906178, 1.193329, 0.619356, 0.807930],
                                               [1.352825, 0.662233, 0.144927, 0.799090],
                                               [0.856150, 0.214590, 5.651497, 1.062401],
                                               [0.194460, 5.660382, 4.761072, 2.471682],
                                               [5.717342, 4.736780, 3.909599, 2.342536]
                                           ]),
                    'expected_output': [93.476, 75.186, 5.2664],
                    'outputs_match_func': q6_check_output,
                    'output_to_str_func': checkutil.list_of_float_to_str,
                    'tries': 20,
                    'matches_required': 12 },
                   )

if __name__ == '__main__':
    checkutil.check( FILL_IN_TEST_CASES, CODE_TEST_CASES, locals() )
