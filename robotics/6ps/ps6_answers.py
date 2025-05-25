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
#       python ps6_answers.py
#
######################################################################

from math import *
import random
import checkutil
from rait import matrix

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

# STUDENT ID
#
# Please specify your GT login ID in the whoami variable (ex: jsmith225).
#

whoami = 'kmccarville3'

# QUESTION 1: MATRIX FILL IN
#
# Fill in the correct Omega Matrix
# Note: the prefilled answers from the video have already been entered for you
# and are not graded
#
Q1_Omega = ([[4, -1,  0, -2,  0], \
                        [-1,  4, -1,  0, -2], \
                        [ 0, -1, 0,  3, -2], \
                        [-2,  0,  0, 2,  0], \
                        [ 0, -2, -2,  0,  4]])

#
# Fill in the correct Xi Matrix
# Each entry should be an integer value
# Note: the prefilled answers from the video have already been entered for you
# and are not graded
#

Q1_Xi = ([[-6],\
                     [-3],\
                     [0],\
                     [0],\
                     [12] ])



# QUESTION 2: ONLINE SLAM

# ------------
# User Instructions
#
# In this problem you will implement a more manageable
# version of graph SLAM in 2 dimensions.
#
# Define a function, online_slam, that takes 5 inputs:
# data, N, num_landmarks, motion_noise, and
# measurement_noise--just as was done in the last
# programming assignment of unit 6. This function
# must return TWO matrices, mu and the final Omega.
#
# Just as with the quiz, your matrices should have x
# and y interlaced, so if there were two poses and 2
# landmarks, Xi would look like:
#
# xi = matrix([[Px0],
#              [Py0],
#              [Px1],
#              [Py1],
#              [Lx0],
#              [Ly0],
#              [Lx1],
#              [Ly1]])
#
# Enter your code in the following function

# --------------------------------
#
# online_slam - retains all landmarks but only most recent robot pose
#

def online_slam(data, N, num_landmarks, motion_noise, measurement_noise):
    '''
    :param data: (list) returned from make_data function
    :param N: (int) the total number of poses (synonymous with number of time steps) that the robot exists in throughout the simulation
    :param num_landmarks: (int) number of landmarks
    :param motion_noise: (float) noise in robot motion
    :param measurement_noise: (float) noise in the measurements
    '''
    #
    #
    ######################## ENTER CODE BELOW HERE #########################
    #
    #

    mu = matrix()
    Omega = matrix()

    return mu, Omega  # make sure you return both of these matrices to be marked correct.

######################## ENTER CODE ABOVE HERE #########################
######################## DO NOT CHANGE CODE BELOW #########################

# -----------
# Testing
#
# You have two methods for testing your code.
#
# 1) You can make your own data with the make_data
#    function. Then you can run it through the
#    provided slam routine and check to see that your
#    online_slam function gives the same estimated
#    final robot pose and landmark positions.
# 2) You can use the solution_check function at the
#    bottom of this document to check your code
#    for the two provided test cases. The grading
#    will be almost identical to this function, so
#    if you pass both test cases, you should be
#    marked correct on the homework.




# ------------------------------------------------
#
# this is the robot class
#
# our robot lives in x-y space, and its motion is
# pointed in a random direction. It moves on a straight line
# until is comes close to a wall at which point it turns
# away from the wall and continues to move.
#
# For measurements, it simply senses the x- and y-distance
# to landmarks. This is different from range and bearing as
# commonly studies in the literature, but this makes it much
# easier to implement the essentials of SLAM without
# cluttered math
#

class robot(object):

    # --------
    # init:
    #   creates robot and initializes location to 0, 0
    #

    def __init__(self, world_size = 100.0, measurement_range = 30.0,
                 motion_noise = 1.0, measurement_noise = 1.0):
        self.measurement_noise = 0.0
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.x = world_size / 2.0
        self.y = world_size / 2.0
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.landmarks = []
        self.num_landmarks = 0


    def rand(self):
        return random.random() * 2.0 - 1.0

    # --------
    #
    # make random landmarks located in the world
    #

    def make_landmarks(self, num_landmarks):
        self.landmarks = []
        for i in range(num_landmarks):
            self.landmarks.append([round(random.random() * self.world_size),
                                   round(random.random() * self.world_size)])
        self.num_landmarks = num_landmarks

    # --------
    #
    # move: attempts to move robot by dx, dy. If outside world
    #       boundary, then the move does nothing and instead returns failure
    #

    def move(self, dx, dy):

        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise

        if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
            return False
        else:
            self.x = x
            self.y = y
            return True

    # --------
    #
    # sense: returns x- and y- distances to landmarks within visibility range
    #        because not all landmarks may be in this range, the list of measurements
    #        is of variable length. Set measurement_range to -1 if you want all
    #        landmarks to be visible at all times
    #

    def sense(self):
        Z = []
        for i in range(self.num_landmarks):
            dx = self.landmarks[i][0] - self.x + self.rand() * self.measurement_noise
            dy = self.landmarks[i][1] - self.y + self.rand() * self.measurement_noise
            if self.measurement_range < 0.0 or abs(dx) + abs(dy) <= self.measurement_range:
                Z.append([i, dx, dy])
        return Z

    # --------
    #
    # print robot location
    #

    def __repr__(self):
        return 'Robot: [x=%.5f y=%.5f]'  % (self.x, self.y)


# ######################################################################

# --------
# this routine makes the robot data
#

def make_data(N, num_landmarks, world_size, measurement_range, motion_noise,
              measurement_noise, distance):
    '''
    :param N: (int) time steps
    :param num_landmarks: (int) number of landmarks
    :param world_size: (float) size of world
    :param measurement_range: (float) range at which we can sense landmarks
    :param motion_noise: (float) noise in robot motion
    :param measurement_noise: (float) noise in the measurements
    :param distance: (float) distance by which robot (intends to) move each
                     iteration
    :returns data: (list) a list of N-1 items.  Each item consists of 2 parts:
                    - Part 1 represents measurement data: list of measurements to landmarks
                        Each measurement contains 3 pieces of data:
                            landmark_id
                            delta_x_distance
                            delta_y_distance
                    - Part 2 represents movement data: list of relative movement distances
                        each movement contains 2 pieces of data:
                            delta_x_distance
                            delta_y_distance
                Example:
                data = [
                    [
                        [
                            [landmark_id, delta_x_distance_meas, delta_y_distance_meas],
                            [...]
                        ],
                        [delta_x_distance_move, delta_y_distance_move]
                    ],
                    [...],
                ]

    '''
    complete = False

    while not complete:

        data = []

        # make robot and landmarks
        r = robot(world_size, measurement_range, motion_noise, measurement_noise)
        r.make_landmarks(num_landmarks)
        seen = [False for row in range(num_landmarks)]

        # guess an initial motion
        orientation = random.random() * 2.0 * pi
        dx = cos(orientation) * distance
        dy = sin(orientation) * distance

        for k in range(N-1):

            # sense
            Z = r.sense()

            # check off all landmarks that were observed
            for i in range(len(Z)):
                seen[Z[i][0]] = True

            # move
            while not r.move(dx, dy):
                # if we'd be leaving the robot world, pick instead a new direction
                orientation = random.random() * 2.0 * pi
                dx = cos(orientation) * distance
                dy = sin(orientation) * distance

            # memorize data
            data.append([Z, [dx, dy]])

        # we are done when all landmarks were observed; otherwise re-run
        complete = (sum(seen) == num_landmarks)

    print(' ')
    print('Landmarks: ', r.landmarks)
    print(r)

    return data

# ######################################################################

# --------------------------------
#
# full_slam - retains entire path and all landmarks
#             Feel free to use this for comparison.
#

def slam(data, N, num_landmarks, motion_noise, measurement_noise):
    '''
    :param data: (list) returned from make_data function
    :param N: (int) the total number of poses (synonymous with number of time steps) that the robot exists in throughout the simulation
    :param num_landmarks: (int) number of landmarks
    :param motion_noise: (float) noise in robot motion
    :param measurement_noise: (float) noise in the measurements
    '''

    # Set the dimension of the filter
    dim = 2 * (N + num_landmarks)

    # make the constraint information matrix and vector
    Omega = matrix()
    Omega.zero(dim, dim)
    Omega.value[0][0] = 1.0
    Omega.value[1][1] = 1.0

    Xi = matrix()
    Xi.zero(dim, 1)
    Xi.value[0][0] = world_size / 2.0
    Xi.value[1][0] = world_size / 2.0

    # process the data

    for k in range(len(data)):

        # n is the index of the robot pose in the matrix/vector
        n = k * 2

        measurement = data[k][0]
        motion      = data[k][1]

        # integrate the measurements
        for i in range(len(measurement)):

            # m is the index of the landmark coordinate in the matrix/vector
            m = 2 * (N + measurement[i][0])

            # update the information maxtrix/vector based on the measurement
            for b in range(2):
                Omega.value[n+b][n+b] +=  1.0 / measurement_noise
                Omega.value[m+b][m+b] +=  1.0 / measurement_noise
                Omega.value[n+b][m+b] += -1.0 / measurement_noise
                Omega.value[m+b][n+b] += -1.0 / measurement_noise
                Xi.value[n+b][0]      += -measurement[i][1+b] / measurement_noise
                Xi.value[m+b][0]      +=  measurement[i][1+b] / measurement_noise


        # update the information maxtrix/vector based on the robot motion
        for b in range(4):
            Omega.value[n+b][n+b] +=  1.0 / motion_noise
        for b in range(2):
            Omega.value[n+b  ][n+b+2] += -1.0 / motion_noise
            Omega.value[n+b+2][n+b  ] += -1.0 / motion_noise
            Xi.value[n+b  ][0]        += -motion[b] / motion_noise
            Xi.value[n+b+2][0]        +=  motion[b] / motion_noise

    # compute best estimate
    mu = Omega.inverse() * Xi

    # return the result
    return mu


# --------------------------------
#
# print the result of SLAM, the robot pose(s) and the landmarks
#

def print_result(N, num_landmarks, result):
    print()
    print('Estimated Pose(s):')
    for i in range(N):
        print('    ['+ ', '.join('%.3f'%x for x in result.value[2*i]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*i+1]) +']')
    print()
    print('Estimated Landmarks:')
    for i in range(num_landmarks):
        print('    ['+ ', '.join('%.3f'%x for x in result.value[2*(N+i)]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*(N+i)+1]) +']')

# ------------------------------------------------------------------------
#
# Main routines
#

num_landmarks      = 5        # number of landmarks
N                  = 20       # time steps
world_size         = 100.0    # size of world
measurement_range  = 50.0     # range at which we can sense landmarks
motion_noise       = 2.0      # noise in robot motion
measurement_noise  = 2.0      # noise in the measurements
distance           = 20.0     # distance by which robot (intends to) move each iteratation


# Uncomment the following three lines to run the full slam routine.

#data = make_data(N, num_landmarks, world_size, measurement_range, motion_noise, measurement_noise, distance)
#result = slam(data, N, num_landmarks, motion_noise, measurement_noise)
#print_result(N, num_landmarks, result)

# Uncomment the following three lines to run the online_slam routine.

#data = make_data(N, num_landmarks, world_size, measurement_range, motion_noise, measurement_noise, distance)
#result = online_slam(data, N, num_landmarks, motion_noise, measurement_noise)
#print_result(1, num_landmarks, result[0])

##########################################################

# ------------
# TESTING
#
# Uncomment one of the test cases below to check that your
# online_slam function works as expected.



# -----------
# Test Case 1

testdata1          = [[[[1, 21.796713239511305, 25.32184135169971], [2, 15.067410969755826, -27.599928007267906]], [16.4522379034509, -11.372065246394495]],
                      [[[1, 6.1286996178786755, 35.70844618389858], [2, -0.7470113490937167, -17.709326161950294]], [16.4522379034509, -11.372065246394495]],
                      [[[0, 16.305692184072235, -11.72765549112342], [2, -17.49244296888888, -5.371360408288514]], [16.4522379034509, -11.372065246394495]],
                      [[[0, -0.6443452578030207, -2.542378369361001], [2, -32.17857547483552, 6.778675958806988]], [-16.66697847355152, 11.054945886894709]]]

answer_mu1         = matrix([[81.63549976607898],
                             [27.175270706192254],
                             [98.09737507003692],
                             [14.556272940621195],
                             [71.97926631050574],
                             [75.07644206765099],
                             [65.30397603859097],
                             [22.150809430682695]])

answer_omega1      = matrix([[0.36603773584905663, 0.0, -0.169811320754717, 0.0, -0.011320754716981133, 0.0, -0.1811320754716981, 0.0],
                             [0.0, 0.36603773584905663, 0.0, -0.169811320754717, 0.0, -0.011320754716981133, 0.0, -0.1811320754716981],
                             [-0.169811320754717, 0.0, 0.6509433962264151, 0.0, -0.05660377358490567, 0.0, -0.40566037735849064, 0.0],
                             [0.0, -0.169811320754717, 0.0, 0.6509433962264151, 0.0, -0.05660377358490567, 0.0, -0.40566037735849064],
                             [-0.011320754716981133, 0.0, -0.05660377358490567, 0.0, 0.6962264150943396, 0.0, -0.360377358490566, 0.0],
                             [0.0, -0.011320754716981133, 0.0, -0.05660377358490567, 0.0, 0.6962264150943396, 0.0, -0.360377358490566],
                             [-0.1811320754716981, 0.0, -0.4056603773584906, 0.0, -0.360377358490566, 0.0, 1.2339622641509433, 0.0],
                             [0.0, -0.1811320754716981, 0.0, -0.4056603773584906, 0.0, -0.360377358490566, 0.0, 1.2339622641509433]])


# -----------
# Test Case 2

testdata2          = [[[[0, 12.637647070797396, 17.45189715769647], [1, 10.432982633935133, -25.49437383412288]], [17.232472057089492, 10.150955955063045]],
                      [[[0, -4.104607680013634, 11.41471295488775], [1, -2.6421937245699176, -30.500310738397154]], [17.232472057089492, 10.150955955063045]],
                      [[[0, -27.157759429499166, -1.9907376178358271], [1, -23.19841267128686, -43.2248146183254]], [-17.10510363812527, 10.364141523975523]],
                      [[[0, -2.7880265859173763, -16.41914969572965], [1, -3.6771540967943794, -54.29943770172535]], [-17.10510363812527, 10.364141523975523]],
                      [[[0, 10.844236516370763, -27.19190207903398], [1, 14.728670653019343, -63.53743222490458]], [14.192077112147086, -14.09201714598981]]]

answer_mu2         = matrix([[63.37479912250136],
                             [78.17644539069596],
                             [61.33207502170053],
                             [67.10699675357239],
                             [62.57455560221361],
                             [27.042758786080363]])

answer_omega2      = matrix([[0.22871751620895048, 0.0, -0.11351536555795691, 0.0, -0.11351536555795691, 0.0],
                             [0.0, 0.22871751620895048, 0.0, -0.11351536555795691, 0.0, -0.11351536555795691],
                             [-0.11351536555795691, 0.0, 0.7867205207948973, 0.0, -0.46327947920510265, 0.0],
                             [0.0, -0.11351536555795691, 0.0, 0.7867205207948973, 0.0, -0.46327947920510265],
                             [-0.11351536555795691, 0.0, -0.46327947920510265, 0.0, 0.7867205207948973, 0.0],
                             [0.0, -0.11351536555795691, 0.0, -0.46327947920510265, 0.0, 0.7867205207948973]])




######################################################################
# Grading methods
#
# Do not modify code below this point.
#
######################################################################

FILL_IN_TEST_CASES = ( {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (0,1),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '6d3fa6b00489c8696734e584c51cdc5b',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (0,2),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '06be207ba1cf756ee326c4018423c12a',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (0,3),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'af96f97da4e522df71f4ded19f7c5204',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (0,4),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '90cf24e4870a76418757479dedf16192',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (1,1),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '4100e8c47d9077063ab904148eb6c79b',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (1,2),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'd9b1c3e5050dcd15977b604868dadc62',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (1,3),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'bf0ec5151ff2fa5363cc53b00330943f',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (1,4),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '9ff970249f16458bb30c065ad27a0ec3',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (2,1),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '3d91bd7424f47102a852752bd1c932b0',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (2,2),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'eaa1a40f2f248e34a6348042ea2bbd84',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (2,3),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'f8d8f9717fb1627278d7295986ca02d3',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (2,4),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '622bb80bcf40bb0ae2f5e5c3475f885b',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (3,1),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '90a963db4576ef5bfc795d17400a7da6',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (3,2),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'f0448872107d35414e14bd3bc7d632c1',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (3,3),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '29369fdc12438a43fcd19b2f20ea12a3',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (3,4),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'b952e7964cea56f44fff262032d9f0cf',
                         'points_avail':      1./17. },
                        {'variable_name': 'Q1_Omega',
                         'variable_idxs':     (4,3),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'f43731b02c724468aa26440531ac0ece',
                         'points_avail':      1./17. },
                        { 'variable_name': 'Q1_Xi',
                         'variable_idxs':     (2,0),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       '9118953504a781f6f9a674aad264d585',
                         'points_avail':      0.5 },
                        { 'variable_name': 'Q1_Xi',
                         'variable_idxs':     (3,0),
                         'str_func':          checkutil.float_to_str,
                         'answer_hash':       'c7d045f9bc2793238281684ce9ddc814',
                         'points_avail':      0.5 } )

# PROGRAMMING


def format_array_output(ans_arrays):

    mu_str = checkutil.format_2d_arr( ans_arrays[0].value )
    omega_str = checkutil.format_2d_arr( ans_arrays[1].value )

    return mu_str + "\n\n" + omega_str

def solution_check(result, answer):

    if len(result) != 2:
        print("Your function must return TWO matrices, mu and Omega")
        return False

    user_mu = result[0]
    user_omega = result[1]

    answer_mu = answer[0]
    answer_omega = answer[1]

    if user_mu.dimx == answer_omega.dimx and user_mu.dimy == answer_omega.dimy:
        print("It looks like you returned your results in the wrong order. Make sure to return mu then Omega.")
        return False

    if user_mu.dimx != answer_mu.dimx or user_mu.dimy != answer_mu.dimy:
        print("Your mu matrix doesn't have the correct dimensions. Mu should be a", answer_mu.dimx, " x ", answer_mu.dimy, "matrix.")
        return False
    else:
        print("Mu has correct dimensions.")

    if user_omega.dimx != answer_omega.dimx or user_omega.dimy != answer_omega.dimy:
        print("Your Omega matrix doesn't have the correct dimensions. Omega should be a", answer_omega.dimx, " x ", answer_omega.dimy, "matrix.")
        return False
    else:
        print("Omega has correct dimensions.")

    # check mu
    ret_val_mu = checkutil.arrays_approx_equal( answer_mu.value, user_mu.value )

    # check omega
    ret_val_omega = checkutil.arrays_approx_equal( answer_omega.value, user_omega.value )

    return ret_val_mu and ret_val_omega


CODE_TEST_CASES = ( { 'function_name': 'online_slam',
                      'function_input': dict( data=testdata1,
                                              N = 5,
                                              num_landmarks = 3,
                                              motion_noise = 2.0,
                                              measurement_noise = 2.0 ),
                      'expected_output': (answer_mu1, answer_omega1),
                      'outputs_match_func': solution_check,
                      'output_to_str_func': format_array_output,
                      'points_avail': 4},
                    { 'function_name': 'online_slam',
                      'function_input': dict( data=testdata2,
                                              N = 6,
                                              num_landmarks = 2,
                                              motion_noise = 3.0,
                                              measurement_noise = 4.0 ),
                      'expected_output': (answer_mu2, answer_omega2),
                      'outputs_match_func': solution_check,
                      'output_to_str_func': format_array_output,
                      'points_avail': 4} )

if __name__ == '__main__':
    checkutil.check( FILL_IN_TEST_CASES, CODE_TEST_CASES, locals() )
