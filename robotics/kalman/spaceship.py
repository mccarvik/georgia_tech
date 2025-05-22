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

# If you see different scores locally and on Gradescope this may be an
# indication that you are uploading a different file than the one you are
# executing locally. If this local ID doesn't match the ID on Gradescope then
# you uploaded a different file.
OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib
    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')


class Spaceship():
    """The Spaceship to guide across the galaxy."""

    def __init__(self, bounds, xy_start):
        """Initialize the Spaceship."""
        self.x_bounds = bounds['x']
        self.y_bounds = bounds['y']
        self.agent_pos_start = xy_start


    def predict_from_observations(self, asteroid_observations):
        """Observe asteroid locations and predict their positions at time t+1.
        Parameters
        ----------
        self = a reference to the current object, the Spaceship
        asteroid_observations = A dictionary in which the keys represent asteroid IDs
        and the values are a dictionary of noisy x-coordinate observations,
        and noisy y-coordinate observations taken at time t.
        asteroid_observations format:
        ```
        `{1: (x-measurement, y-measurement),
          2: (x-measurement, y-measurement)...
          100: (x-measurement, y-measurement),
          }`
        ```

        Returns
        -------
        The output of the `predict_from_observations` function should be a dictionary of tuples
        of estimated asteroid locations one timestep into the future
        (i.e. the inputs are for measurements taken at time t, and you return where the asteroids will be at time t+1).

        A dictionary of tuples containing i: (x, y), where i, x, and y are:
        i = the asteroid's ID
        x = the estimated x-coordinate of asteroid i's position for time t+1
        y = the estimated y-coordinate of asteroid i's position for time t+1
        Return format:
        `{1: (x-coordinate, y-coordinate),
          2: (x-coordinate, y-coordinate)...
          100: (x-coordinate, y-coordinate)
          }`
        """
        # To view the visualization with the default pdf output (incorrect) uncomment the line below
        # return asteroid_observations

        # FOR STUDENT TODO: Update the Spaceship's estimate of where the asteroids will be located in the next time step

        return {-1: (5.5, 5.5)}

    def jump(self, asteroid_observations, agent_data):
        """ Return the id of the asteroid the spaceship should jump/hop onto in the next timestep
        ----------
        self = a reference to the current object, the Spaceship
        asteroid_observations: Same as predict_from_observations method
        agent_data: a dictionary containing agent related data:
        'jump_distance' - a float representing agent jumping distance,
        'ridden_asteroid' - an int representing the ID of the ridden asteroid if available, None otherwise.
        Note: 'agent_pos_start' - A tuple representing the (x, y) position of the agent at t=0 is available in the constructor.

        agent_data format:
        {'ridden_asteroid': None,
         'jump_distance': agent.jump_distance,
         }
        Returns
        -------
        You are to return two items.
        1: idx, this represents the ID of the asteroid on which to jump if a jump should be performed in the next timestep.
        Return None if you do not intend to jump on an asteroid in the next timestep
        2. Return the estimated positions of the asteroids (i.e. the output of 'predict_from_observations method)
        IFF you intend to have them plotted in the visualization. Otherwise return None
        -----
        an example return
        idx to hop onto in the next timestep: 3,
        estimated_results = {1: (x-coordinate, y-coordinate),
          2: (x-coordinate, y-coordinate)}

        return 3, estimated_return

        """
        # FOR STUDENT TODO: Update the idx of the asteroid on which to jump
        idx = False

        return idx, None

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
