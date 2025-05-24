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

import math

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

    
    # I added all the matrix opertaions here just to make it easier later on
    # its all standard matrix math / linear algebra

    # I consulted LLMs for math and theoretical help
    # namely claude 3.5 sonnet
    # All work is my own, no code was copy and pasted and no code was taken from other sources such as stack overflow or github

    def matrix_multiply(self, A, B):
        """Multiply two matrices A and B."""
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def matrix_transpose(self, A):
        """Transpose a matrix A."""
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    def matrix_inverse(self, A):
        """Inverse of a 2x2 matrix."""
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if det == 0:
            raise ValueError("Matrix is singular")
        return [
            [A[1][1]/det, -A[0][1]/det],
            [-A[1][0]/det, A[0][0]/det]
        ]

    def matrix_subtract(self, A, B):
        """Subtract matrix B from matrix A."""
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def matrix_add(self, A, B):
        """Add two matrices A and B."""
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


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
        
        # go thru each asteroid in the observations
        for asteroid_id, measurement in asteroid_observations.items():

            # Initialize Kalman filter if we dont have on yet for this asteroid
            if not hasattr(self, f'kf_{asteroid_id}'):
                # State vector: [x, y, vx, vy, ax, ay]
                x, y = measurement
                initial_state = [x, y, 0, 0, 0, 0]
                
                # Initial covariance matrix
                # P Matrix of the filter
                initial_P = [[1000 if i == j else 0 for j in range(6)] for i in range(6)]
                
                # Process noise covariance
                # Q Matrix of the filter
                Q = [[0.1 if i == j else 0 for j in range(6)] for i in range(6)]
                
                # Measurement noise covariance
                # R Matrix of the filter
                # USED A LOT of trial and error here, these seem to be the best values
                r_x = 10000
                r_y = 10000
                R = [
                    [r_x, 0],      # variance in x-direction
                    [0, r_y]       # variance in y-direction
                ]
                
                # Measurement matrix (we only measure position)
                # H Matrix of the filter
                H = [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]
                ]
                
                # Store Kalman filter parameters
                # we now have a kalman filter for each asteroid
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
            dt = 1.0  # Each timestep is 1 second
            
            # State transition matrix
            # F Matrix of the filter
            # where all the magic happens
            F = [
                [1, 0, dt, 0, 0.5*dt**2, 0],
                [0, 1, 0, dt, 0, 0.5*dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ]
            
            # Predict state
            # start going thru calculations for the velocity and acceleration
            kf['state'] = self.matrix_multiply(F, [[x] for x in kf['state']])
            kf['state'] = [row[0] for row in kf['state']]
            
            # Predict covariance
            # this cam straight from the pdf
            F_P = self.matrix_multiply(F, kf['P'])
            F_P_FT = self.matrix_multiply(F_P, self.matrix_transpose(F))
            kf['P'] = self.matrix_add(F_P_FT, kf['Q'])
            
            # Update step
            # Measurement
            z = [[measurement[0]], [measurement[1]]]
            
            # Innovation
            # mor copying from the PDF
            H_state = self.matrix_multiply(kf['H'], [[x] for x in kf['state']])
            y = self.matrix_subtract(z, H_state)
            
            # Innovation covariance
            # going thru all the kalman filter math here
            H_P = self.matrix_multiply(kf['H'], kf['P'])
            H_P_HT = self.matrix_multiply(H_P, self.matrix_transpose(kf['H']))
            S = self.matrix_add(H_P_HT, kf['R'])
            
            # Kalman gain
            # finally we get the kalman gain
            P_HT = self.matrix_multiply(kf['P'], self.matrix_transpose(kf['H']))
            S_inv = self.matrix_inverse(S)
            K = self.matrix_multiply(P_HT, S_inv)
            
            # Update state
            # and now we get to the part where we update the state
            K_y = self.matrix_multiply(K, y)
            kf['state'] = [kf['state'][i] + K_y[i][0] for i in range(6)]
            
            # Update covariance
            # update the covariance matrix as well for next iteration
            K_H = self.matrix_multiply(K, kf['H'])
            I_KH = [[1 if i == j else 0 for j in range(6)] for i in range(6)]
            for i in range(6):
                for j in range(6):
                    I_KH[i][j] -= K_H[i][j]
            kf['P'] = self.matrix_multiply(I_KH, kf['P'])
            
            # Predict next position
            # this is the final step, we get the predicted position for the next timestep
            next_state = self.matrix_multiply(F, [[x] for x in kf['state']])
            predicted_positions[asteroid_id] = (next_state[0][0], next_state[1][0])
        
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
        # hopefully we did a goood job above
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
        
        # Use 70% of max jump distance as safety margin
        # lots of trial and error here, this is the best I could get
        max_jump_distance = agent_data['jump_distance'] * 0.70
        
        # Calculate screen center (x-axis only)
        # this is important for later
        center_x = (self.x_bounds[0] + self.x_bounds[1]) / 2
        
        # go thru each asteroid in the predicted positions
        for asteroid_id, pred_pos in predicted_positions.items():
            # Skip if we're already on this asteroid
            if asteroid_id == agent_data['ridden_asteroid']:
                continue
            
            # Get Kalman filter state for this asteroid
            kf = self.__getattribute__(f'kf_{asteroid_id}')
            
            # Get velocity components
            vx = kf['state'][2]  # x-velocity
            vy = kf['state'][3]  # y-velocity
            
            # Skip if not moving north (positive y-velocity)
            # this was crucial for early jumps of the map
            if vy <= 0.001:
                continue
            
            # Calculate distance between current position and predicted asteroid position
            current_x, current_y = current_pos
            pred_x, pred_y = pred_pos
            
            # Skip if asteroid is not above current position
            # not totally necessary but just ended up being safer
            if pred_y <= current_y * 0.98:
                continue
            
            # Calculate Euclidean distance
            dx = pred_x - current_x
            dy = pred_y - current_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Skip if too far or outside bounds
            if dist > max_jump_distance:
                continue
            if pred_x < self.x_bounds[0] or pred_x > self.x_bounds[1]:
                continue
            # big edit here with the 1.01 to make sure we dont jump off the map with some uncertainty baked in
            if pred_y < self.y_bounds[0] or pred_y*1.05 > self.y_bounds[1]:
                continue
            
            # Calculate uncertainty based on position covariance
            position_uncertainty = math.sqrt(kf['P'][0][0] + kf['P'][1][1])
            uncertainty_penalty = position_uncertainty * dist
            
            # Calculate distance to center (x-axis only)
            dist_to_center_x = abs(pred_x - center_x)
            
            # Score components:
            # 1. Vertical progress (higher is better)
            vertical_score = pred_y * 2.0  # Reduced weight
            
            # 2. Distance penalty (closer is better)
            distance_penalty = dist
            
            # 3. Uncertainty penalty
            uncertainty_penalty = position_uncertainty * dist
            
            # 4. Velocity bonuses
            north_velocity_bonus = vy * 1.5  # Reduced weight
            center_velocity_bonus = 0
            # This was HUGE. Jump on asteroids that are moving towards the center of the map
            if pred_x > center_x and vx < 0:  # Moving towards center from right
                center_velocity_bonus = abs(vx) * 5.0  # Increased weight
            elif pred_x < center_x and vx > 0:  # Moving towards center from left
                center_velocity_bonus = abs(vx) * 5.0  # Increased weight
            
            # 5. Center position bonus (prefer being closer to center on x-axis)
            center_position_bonus = 2000 / (1 + dist_to_center_x)  # Doubled weight
            
            # Calculate final score
            score = (vertical_score - 
                    distance_penalty - 
                    uncertainty_penalty + 
                    north_velocity_bonus + 
                    center_velocity_bonus + 
                    center_position_bonus)
            
            if score > best_score:
                best_score = score
                best_asteroid = asteroid_id
        
        return best_asteroid, predicted_positions


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
