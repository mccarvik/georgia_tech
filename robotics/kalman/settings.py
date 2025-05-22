import math

# -------------------------------------------------
SUCCESS = 'Success!'
FAILURE_LOW_SCORE = 'Failed: Min Score Not Met'
FAILURE_UNIMPLEMENTED = 'Please Implement Estimation'
NAV_FAILURE = 'UNKNOWN FAILURE'  # Custom failure states for defense.
NUM_TIMESTEPS_TO_PREDICT_INFUTURE = 1
FAILURE_AGENT_OUT_OF_BOUNDS = 'Failed: Agent Out of Bounds'  # Custom failure states for defense.
FAILURE_ASTEROID_OUT_OF_JUMP_RANGE = 'Failed: Jump Range Exceeded'  # Custom failure states for defense.
FAILURE_TIME_STEP_LIMIT_EXCEEDED = 'Failed: Time step limit Exceeded'  # Custom failure states for defense.
DISPLAY_ASTEROID_MATCH_RANGE = True
DISPLAY_ORIGIN_TO_DESTINATION_LINE = False
DISPLAY_ESTIMATED_SPACESHIP_MATCH_RANGE = True
LIGHT_BLUE_COLOR = '#6775d5'

runner_settings = {
                   'SUCCESS' : SUCCESS,
                   'FAILURE_LOW_SCORE': FAILURE_LOW_SCORE,
                   'NAV_FAILURE': NAV_FAILURE,
                   'FAILURE_UNIMPLEMENTED': FAILURE_UNIMPLEMENTED,
                   'FAILURE_TIME_STEP_LIMIT_EXCEEDED': FAILURE_TIME_STEP_LIMIT_EXCEEDED,
                   'FAILURE_AGENT_OUT_OF_BOUNDS': FAILURE_AGENT_OUT_OF_BOUNDS,
                   'NUM_TIMESTEPS_TO_PREDICT_INFUTURE': NUM_TIMESTEPS_TO_PREDICT_INFUTURE,
                   'FAILURE_ASTEROID_OUT_OF_JUMP_RANGE': FAILURE_ASTEROID_OUT_OF_JUMP_RANGE,
                   'DISPLAY_ASTEROID_MATCH_RANGE': DISPLAY_ASTEROID_MATCH_RANGE,
                   'DISPLAY_ESTIMATED_SPACESHIP_MATCH_RANGE': DISPLAY_ESTIMATED_SPACESHIP_MATCH_RANGE,
                   'DISPLAY_ORIGIN_TO_DESTINATION_LINE': DISPLAY_ORIGIN_TO_DESTINATION_LINE,
                   'COLOR': LIGHT_BLUE_COLOR,
                   }

# Settings related to asteroids -------------------------------------------------
FIELD_X_BOUNDS = (0, 2)
FIELD_Y_BOUNDS = (0, 2)
IN_BOUNDS = {"x_bounds": FIELD_X_BOUNDS, "y_bounds": FIELD_Y_BOUNDS}
SLEEP_CONSTANT = 0.0
asteroid_attributes = {'size': ['small', 'medium', 'large'],
                        'color': ['red', 'green', 'blue'],
                        'form': ['fire', 'ice', 'rock']}

generate_asteroid_settings = {'FIELD_X_BOUNDS': FIELD_X_BOUNDS,
                               'FIELD_Y_BOUNDS': FIELD_Y_BOUNDS,
                               'IN_BOUNDS': IN_BOUNDS,
                               'asteroid_attributes': asteroid_attributes,
                               }

# settings related to visualization -------------------------------------------------
# Set DEBUG_DISPLAY to True to see the asteroid ID numbers and the labels of
# the corners of the arena in the GUI, or set it to FALSE not show those
# values
DEBUG_DISPLAY = True
DARK_MODE = True

# DARK_MODE = False
LIGHT_COLOR_PALETTE = {
    'agent': 'blue',
    'agent_estimated_correct': 'green',
    'agent_estimated_incorrect': 'red',
    'spaceship':    'black',
    'asteroid': 'grey60',
    'asteroid_range': 'red',
    'asteroid_text': '',
    'estimated_asteroid_loc_correct': '#88ff88', # mint green
    'estimated_asteroid_loc_incorrect': '#aa4444', # deep pink
    'border': 'black',
    'border_goal': 'green',
    'label_text': 'black',
    'background_color': 'white',
    'toggle_circle1': 'black',
    'toggle_fill_circle1': 'grey',
    'toggle_circle2': 'grey60',
    'toggle_fill_circle2': 'grey60',
    'rect_small': 'grey60',
}
DARK_COLOR_PALETTE = {
    'agent': 'blue',
    # 'agent': 'yellow',
    'agent_estimated_correct': 'green',
    'agent_estimated_incorrect': 'red',
    'spaceship': 'green',
    'asteroid': 'grey60',
    'asteroid_range': '#ad6367',
    'estimated_asteroid_loc_correct': 'green', # mint green
    'estimated_asteroid_loc_incorrect': '#ad6367', # deep pink
    'border': 'white',
    'border_goal': 'green',
    'label_text': 'white',
    'background_color': 'grey25', # grey0 = black, grey100 = white
    'toggle_circle1': '',
    'toggle_fill_circle1': 'green',
    'toggle_circle2': 'black',
    'toggle_fill_circle2': 'grey',
    'rect_small': 'green',
}


COLOR_PALETTE = DARK_COLOR_PALETTE if DARK_MODE else LIGHT_COLOR_PALETTE


visualizer_settings = {'DEBUG_DISPLAY': DEBUG_DISPLAY,
                       'DARK_MODE': DARK_MODE,
                       'LIGHT_COLOR_PALETTE': LIGHT_COLOR_PALETTE,
                       'DARK_COLOR_PALETTE': DARK_COLOR_PALETTE,
                       'COLOR_PALETTE': COLOR_PALETTE,
                       'SLEEP_CONSTANT': SLEEP_CONSTANT,
                       }

# -------------------------------------------------


ESTIMATE_TIMEOUT = 200
JUMP_TIMEOUT = 50
FAILURE_EXCEPTION = 'exception_raised'
FAILURE_TIMEOUT = 'execution_time_exceeded'

# This flag is used to check whether project files listed in the json have been modified.
# Modifications include (but are not limited to) print statements, changing flag values, etc.
# If you have modified the project files in some way, the results may not be accurate.
# Turn file_checker on by setting the flag to True to ensure you are running against
# the same framework as the Gradescope autograder.
file_checker = False  # set to True to turn file checking on

test_settings = {
                 'ESTIMATE_TIMEOUT': ESTIMATE_TIMEOUT,
                 'JUMP_TIMEOUT': JUMP_TIMEOUT,
                 'FAILURE_EXCEPTION': FAILURE_EXCEPTION,
                 'FAILURE_TIMEOUT': FAILURE_TIMEOUT,
                 'FILE_CHECKER': file_checker}
