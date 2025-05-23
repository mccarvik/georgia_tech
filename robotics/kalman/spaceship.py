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

import numpy as np

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
        # I worked with LLMs for math questions as well as theoretical issues, all code is my own and exclustively written by me
        # Primary LLM used : Claude 3.5 Sonnet
        # return {-1: (5.5, 5.5)}

        predicted_positions = {}
        
        for asteroid_id, measurement in asteroid_observations.items():
            # Initialize state if not exists
            if not hasattr(self, f'state_{asteroid_id}'):
                print(f"Setting up initial state for asteroid {asteroid_id}")
                x, y = measurement
                # Initialize with first measurement
                self.__setattr__(f'state_{asteroid_id}', {
                    't': 0,
                    'measurements': [(x, y)],  # Store all measurements
                    'cposx': x,  # Initial position coefficient
                    'cposy': y,
                    'cvelx': 0,  # Initial velocity coefficient
                    'cvely': 0,
                    'caccx': 0,  # Initial acceleration coefficient
                    'caccy': 0
                })
            
            # Get current state
            state = self.__getattribute__(f'state_{asteroid_id}')
            
            # Update time and store measurement
            state['t'] += 1
            state['measurements'].append(measurement)
            
            # Need at least 3 measurements to calculate coefficients
            if len(state['measurements']) >= 3:
                # Get last 3 measurements
                m1 = state['measurements'][-3]  # t-2
                m2 = state['measurements'][-2]  # t-1
                m3 = state['measurements'][-1]  # t
                
                # Calculate coefficients using the last 3 measurements
                # For x coordinates:
                # m1 = cposx + cvelx*(t-2) + (1/2)*caccx*(t-2)^2
                # m2 = cposx + cvelx*(t-1) + (1/2)*caccx*(t-1)^2
                # m3 = cposx + cvelx*t + (1/2)*caccx*t^2
                
                # Solve for acceleration coefficient first
                # a = (m3 - 2*m2 + m1) / (t^2 - 2*(t-1)^2 + (t-2)^2)
                t = state['t']
                denom = t**2 - 2*(t-1)**2 + (t-2)**2
                
                if denom != 0:  # Avoid division by zero
                    # Calculate acceleration coefficients
                    state['caccx'] = (m3[0] - 2*m2[0] + m1[0]) / denom
                    state['caccy'] = (m3[1] - 2*m2[1] + m1[1]) / denom
                    
                    # Calculate velocity coefficients
                    # v = (m2 - m1 - (1/2)*a*((t-1)^2 - (t-2)^2)) / ((t-1) - (t-2))
                    state['cvelx'] = (m2[0] - m1[0] - 0.5*state['caccx']*((t-1)**2 - (t-2)**2))
                    state['cvely'] = (m2[1] - m1[1] - 0.5*state['caccy']*((t-1)**2 - (t-2)**2))
                    
                    # Calculate position coefficients
                    # p = m1 - v*(t-2) - (1/2)*a*(t-2)^2
                    state['cposx'] = m1[0] - state['cvelx']*(t-2) - 0.5*state['caccx']*(t-2)**2
                    state['cposy'] = m1[1] - state['cvely']*(t-2) - 0.5*state['caccy']*(t-2)**2
            
            # Predict next position using the motion model
            # x(t+1) = cposx + cvelx*(t+1) + (1/2)*caccx*(t+1)^2
            next_x = state['cposx'] + state['cvelx']*(state['t'] + 1) + 0.5*state['caccx']*(state['t'] + 1)**2
            next_y = state['cposy'] + state['cvely']*(state['t'] + 1) + 0.5*state['caccy']*(state['t'] + 1)**2
            
            predicted_positions[asteroid_id] = (next_x, next_y)
            
            print(f"Asteroid {asteroid_id} at t={state['t']}:")
            print(f"  Current: {measurement}")
            print(f"  Predicted: ({next_x}, {next_y})")
            print(f"  Coefficients: pos=({state['cposx']}, {state['cposy']}), vel=({state['cvelx']}, {state['cvely']}), acc=({state['caccx']}, {state['caccy']})")
        
        return predicted_positions


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
        # idx = False
        # return idx, None

        # Get predicted positions
        predicted_positions = self.predict_from_observations(asteroid_observations)
        
        # Get current position
        if agent_data['ridden_asteroid'] is not None:
            # When riding an asteroid, use its predicted position (best estimate of true position)
            current_pos = predicted_positions[agent_data['ridden_asteroid']]
        else:
            # At start, we know our true position exactly
            current_pos = self.agent_pos_start
        
        # Find best asteroid to jump to
        best_asteroid = None
        best_score = float('-inf')
        
        # Use 85% of max jump distance as safety margin to account for noise
        max_jump_distance = agent_data['jump_distance'] * 0.85
        
        for asteroid_id, pred_pos in predicted_positions.items():
            # Skip if we're already on this asteroid
            if asteroid_id == agent_data['ridden_asteroid']:
                continue
            
            # Calculate distance between current position and predicted asteroid position
            current_x, current_y = current_pos
            pred_x, pred_y = pred_pos
            
            # Calculate Euclidean distance
            dx = pred_x - current_x
            dy = pred_y - current_y
            dist = (dx*dx + dy*dy)**0.5
            
            # Skip if too far or outside bounds
            if dist > max_jump_distance:
                continue
            if pred_x < self.x_bounds[0] or pred_x > self.x_bounds[1]:
                continue
            if pred_y < self.y_bounds[0] or pred_y > self.y_bounds[1]:
                continue
            
            # Get state for this asteroid
            state = self.__getattribute__(f'state_{asteroid_id}')
            
            # Score based on:
            # 1. Vertical progress (higher is better)
            # 2. Distance (closer is better)
            # 3. Uncertainty (lower is better)
            # 4. Velocity direction (prefer upward movement)
            vertical_weight = 3.0  # Prioritize vertical progress
            
            # Calculate uncertainty based on number of measurements
            # More measurements = more confidence in our prediction
            measurement_confidence = min(1.0, len(state['measurements']) / 10.0)  # Cap at 1.0
            uncertainty_penalty = (1.0 - measurement_confidence) * dist
            
            # Bonus for upward movement
            velocity_bonus = max(0, state['cvely']) * 0.5
            
            # Calculate score with noise consideration
            score = (pred_y * vertical_weight) - dist - uncertainty_penalty + velocity_bonus
            
            if score > best_score:
                best_score = score
                best_asteroid = asteroid_id
                
            print(f"Current position: ({current_x}, {current_y})")
            print(f"Asteroid {asteroid_id} position: ({pred_x}, {pred_y})")
            print(f"Best asteroid: {best_asteroid}, Best score: {best_score}")
            print(f"Asteroid {asteroid_id} distance: {dist}, Max jump distance: {max_jump_distance}")
            print(f"Measurement confidence: {measurement_confidence}, Uncertainty penalty: {uncertainty_penalty}")
        
        return best_asteroid, predicted_positions


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
