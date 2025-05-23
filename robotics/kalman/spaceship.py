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
        predicted_positions = {}
        
        for asteroid_id, measurement in asteroid_observations.items():
            # Initialize Kalman filter if not exists
            if not hasattr(self, f'kf_{asteroid_id}'):
                # State vector: [x, y, vx, vy, ax, ay]
                # Initial state
                x, y = measurement
                initial_state = np.array([x, y, 0, 0, 0, 0])
                
                # Initial covariance matrix
                initial_P = np.eye(6) * 1000  # High initial uncertainty
                
                # Process noise covariance
                Q = np.eye(6) * 0.1
                
                # Measurement noise covariance
                R = np.eye(2) * 0.1
                
                # Measurement matrix (we only measure position)
                H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0]])
                
                # Store Kalman filter parameters
                self.__setattr__(f'kf_{asteroid_id}', {
                    'state': initial_state,
                    'P': initial_P,
                    'Q': Q,
                    'R': R,
                    'H': H,
                    't': 0
                })
            
            # Get Kalman filter parameters
            kf = self.__getattribute__(f'kf_{asteroid_id}')
            
            # Update time
            kf['t'] += 1
            t = kf['t']
            
            # Prediction step
            # For constant acceleration model:
            # x(t+1) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            # v(t+1) = v(t) + a(t)*dt
            # a(t+1) = a(t)
            dt = 1.0  # Each timestep is 1 second
            
            # State transition matrix
            F = np.array([
                [1, 0, dt, 0, 0.5*dt**2, 0],
                [0, 1, 0, dt, 0, 0.5*dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            
            # Predict state
            kf['state'] = F @ kf['state']
            
            # Predict covariance
            kf['P'] = F @ kf['P'] @ F.T + kf['Q']
            
            # Update step
            # Measurement
            z = np.array(measurement)
            
            # Innovation
            y = z - kf['H'] @ kf['state']
            
            # Innovation covariance
            S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']
            
            # Kalman gain
            K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)
            
            # Update state
            kf['state'] = kf['state'] + K @ y
            
            # Update covariance
            kf['P'] = (np.eye(6) - K @ kf['H']) @ kf['P']
            
            # Predict next position
            next_state = F @ kf['state']
            predicted_positions[asteroid_id] = (next_state[0], next_state[1])
            
            # print(f"Asteroid {asteroid_id} at t={t}:")
            # print(f"  Current: {measurement}")
            # print(f"  Predicted: ({next_state[0]}, {next_state[1]})")
            # print(f"  State: {kf['state']}")
        
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
        # Get predicted positions
        predicted_positions = self.predict_from_observations(asteroid_observations)
        
        # Get current position
        if agent_data['ridden_asteroid'] is not None:
            # When riding an asteroid, use its predicted position
            current_pos = predicted_positions[agent_data['ridden_asteroid']]
        else:
            # At start, we know our true position exactly
            current_pos = self.agent_pos_start
        
        # Find best asteroid to jump to
        best_asteroid = None
        best_score = float('-inf')
        
        # Use 85% of max jump distance as safety margin
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
            
            # Get Kalman filter state for this asteroid
            kf = self.__getattribute__(f'kf_{asteroid_id}')
            
            # Score based on:
            # 1. Vertical progress (higher is better)
            # 2. Distance (closer is better)
            # 3. Uncertainty (lower is better)
            # 4. Velocity direction (prefer upward movement)
            vertical_weight = 3.0  # Prioritize vertical progress
            
            # Calculate uncertainty based on position covariance
            position_uncertainty = np.sqrt(kf['P'][0,0] + kf['P'][1,1])
            uncertainty_penalty = position_uncertainty * dist
            
            # Bonus for upward movement
            velocity_bonus = max(0, kf['state'][3]) * 0.5  # y-velocity
            
            # Calculate score
            score = (pred_y * vertical_weight) - dist - uncertainty_penalty + velocity_bonus
            
            if score > best_score:
                best_score = score
                best_asteroid = asteroid_id
        
        return best_asteroid, predicted_positions


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
