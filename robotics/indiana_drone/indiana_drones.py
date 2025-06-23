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

"""
 === Introduction ===

   The assignment is broken up into two parts.

   Part A:
        Create a SLAM implementation to process a series of landmark measurements (location of tree centers) and movement updates.
        The movements are defined for you so there are no decisions for you to make, you simply process the movements
        given to you.
        Hint: A planner with an unknown number of motions works well with an online version of SLAM.

    Part B:
        Here you will create the action planner for the drone.  The returned actions will be executed with the goal being to navigate to
        and extract the treasure from the environment marked by * while avoiding obstacles (trees).
        Actions:
            'move distance steering'
            'extract treasure_type x_coordinate y_coordinate'
        Example Actions:
            'move 1 1.570963'
            'extract * 1.5 -0.2'

    Note: All of your estimates should be given relative to your drone's starting location.

    Details:
    - Start position
      - The drone will land at an unknown location on the map, however, you can represent this starting location
        as (0,0), so all future drone location estimates will be relative to this starting location.
    - Measurements
      - Measurements will come from trees located throughout the terrain.
        * The format is {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'D', 'radius':0.5}, ...}
      - Only trees that are within the horizon distance will return measurements. Therefore new trees may appear as you move through the environment.
    - Movements
      - Action: 'move 1.0 1.570963'
        * The drone will turn counterclockwise 90 degrees [1.57 radians] first and then move 1.0 meter forward.
      - Movements are stochastic due to, well, it being a robot.
      - If max distance or steering is exceeded, the drone will not move.
      - Action: 'extract * 1.5 -0.2'
        * The drone will attempt to extract the specified treasure (*) from the current location of the drone (1.5, -0.2).
      - The drone must be within 0.25 distance to successfully extract a treasure.

    The drone will always execute a measurement first, followed by an action.
    The drone will have a time limit of 10 seconds to find and extract all of the needed treasures.
"""

# NOTE:
# I used LLMs to help me with the theory on this assignment. Notably anthropic Claude 3.5 Sonnet.
# I did not use any code and nothing was copy and pasted. All work is my own.


from typing import Dict, List
import math
import heapq

# If you see different scores locally and on Gradescope this may be an indication
# that you are uploading a different file than the one you are executing locally.
# If this local ID doesn't match the ID on Gradescope then you uploaded a different file.
OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib
    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')

class SLAM:
    """Create a basic SLAM module.
    """

    def __init__(self):
        """Initialize SLAM components here.
        """
        # Initialize drone position at (0,0) with bearing 0
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_bearing = 0.0
        
        # Dictionary to store landmark positions
        self.landmarks = {}
        
        # Noise parameters for measurements and movements
        self.measure_distance_noise = 0.05
        self.measure_bearing_noise = 0.03
        self.move_distance_noise = 0.05
        self.move_steering_noise = 0.02
        
        # Uncertainty parameters
        self.drone_uncertainty = 0.1
        self.landmark_uncertainty = 0.2
        
        # Movement history for GraphSLAM
        self.movement_history = []
        self.measurement_history = []

    # Provided Functions
    def get_coordinates(self):
        """
        Retrieves the estimated (x, y) locations in meters of the drone and all landmarks (trees) when called.

        Args: None

        Returns:
            The (x,y) coordinates in meters of the drone and all landmarks (trees) in the format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        coordinates = {'self': (self.drone_x, self.drone_y)}
        coordinates.update(self.landmarks)
        return coordinates

    def process_measurements(self, measurements: Dict):
        """
        Process a new series of measurements and update (x,y) location of drone and landmarks

        Args:
            measurements: Collection of measurements of tree positions and radius
                in the format {'landmark id':{'distance': float <meters>, 'bearing':float <radians>, 'type': char, 'radius':float <meters>}, ...}

        """
        for landmark_id, measurement in measurements.items():
            distance = measurement['distance']
            bearing = measurement['bearing']
            landmark_type = measurement['type']
            radius = measurement['radius']
            
            # Convert polar coordinates (distance, bearing) to Cartesian coordinates
            # bearing is relative to drone's current orientation
            absolute_bearing = self.drone_bearing + bearing
            
            # Calculate landmark position relative to drone
            landmark_x = self.drone_x + distance * math.cos(absolute_bearing)
            landmark_y = self.drone_y + distance * math.sin(absolute_bearing)
            
            # Update landmark position with uncertainty handling
            if landmark_id in self.landmarks:
                # Update existing landmark with weighted average
                old_x, old_y = self.landmarks[landmark_id]
                weight_old = 1.0 / self.landmark_uncertainty
                weight_new = 1.0 / (distance * 0.1)  # Uncertainty based on distance
                
                total_weight = weight_old + weight_new
                landmark_x = (weight_old * old_x + weight_new * landmark_x) / total_weight
                landmark_y = (weight_old * old_y + weight_new * landmark_y) / total_weight
                
                # Reduce uncertainty for this landmark
                self.landmark_uncertainty = max(0.05, self.landmark_uncertainty * 0.9)
            else:
                # New landmark - initialize with higher uncertainty
                self.landmark_uncertainty = 0.3
            
            # Update or add landmark position
            self.landmarks[landmark_id] = (landmark_x, landmark_y)
        
        # Store measurement for potential GraphSLAM optimization
        self.measurement_history.append(measurements)

    def process_movement(self, distance: float, steering: float):
        """
        Process a new movement and update (x,y) location of drone

        Args:
            distance: distance to move in meters
            steering: amount to turn in radians
        """
        # Store movement for potential GraphSLAM optimization
        self.movement_history.append((distance, steering))
        
        # Update bearing first (turn then move)
        self.drone_bearing += steering
        
        # Normalize bearing to [-pi, pi]
        self.drone_bearing = ((self.drone_bearing + math.pi) % (2 * math.pi)) - math.pi
        
        # Update position
        self.drone_x += distance * math.cos(self.drone_bearing)
        self.drone_y += distance * math.sin(self.drone_bearing)
        
        # Increase uncertainty with movement
        self.drone_uncertainty += 0.01
        
        # Periodically run GraphSLAM optimization
        if len(self.movement_history) % 10 == 0:
            self._optimize_graph()
    
    def _optimize_graph(self):
        """Simple GraphSLAM optimization to improve estimates."""
        # This is a simplified version - in practice, you'd use a full GraphSLAM implementation
        # For now, we'll just do some basic smoothing
        
        if len(self.movement_history) < 2:
            return
        
        # Simple smoothing of landmark positions based on consistency
        for landmark_id in self.landmarks:
            # Check if landmark appears in multiple measurements
            appearances = 0
            total_x = 0
            total_y = 0
            
            for measurements in self.measurement_history:
                if landmark_id in measurements:
                    appearances += 1
                    measurement = measurements[landmark_id]
                    # Recalculate position based on this measurement
                    # (This is simplified - you'd need to track drone position at each measurement)
                    total_x += self.landmarks[landmark_id][0]
                    total_y += self.landmarks[landmark_id][1]
            
            if appearances > 1:
                # Average the positions
                avg_x = total_x / appearances
                avg_y = total_y / appearances
                self.landmarks[landmark_id] = (avg_x, avg_y)


class IndianaDronesPlanner:
    """
    Create a planner to navigate the drone to reach and extract the treasure marked by * from an unknown start position while avoiding obstacles (trees).
    """

    def __init__(self, max_distance: float, max_steering: float):
        """
        Initialize your planner here.

        Args:
            max_distance: the max distance the drone can travel in a single move in meters.
            max_steering: the max steering angle the drone can turn in a single move in radians.
        """
        self.max_distance = max_distance
        self.max_steering = max_steering
        
        # Create SLAM instance for accurate position tracking
        self.slam = SLAM()
        
        # Store landmarks and their positions (will be updated by SLAM)
        self.landmarks = {}
        
        # Track last movement to update SLAM
        self.last_distance = 0.0
        self.last_steering = 0.0
        
        # Path planning state
        self.path = []
        self.current_path_index = 0
        self.target_reached = False
        
        # Grid parameters for pathfinding
        self.grid_resolution = 0.1
        self.grid_bounds = (-10, 10)  # Will be expanded as needed

    def next_move(self, measurements: Dict, treasure_location: Dict):
        """Next move based on the current set of measurements.

        Args:
            measurements: Collection of measurements of tree positions and radius in the format
                          {'landmark id':{'distance': float <meters>, 'bearing':float <radians>, 'type': char, 'radius':float <meters>}, ...}
            treasure_location: Location of Treasure in the format {'x': float <meters>, 'y':float <meters>, 'type': char '*'}

        Return: action: str, points_to_plot: dict [optional]
            action (str): next command to execute on the drone.
                allowed:
                    'move distance steering'
                    'move 1.0 1.570963'  - Turn left 90 degrees and move 1.0 distance.

                    'extract treasure_type x_coordinate y_coordinate'
                    'extract * 1.5 -0.2' - Attempt to extract the treasure * from your current location (x = 1.5, y = -0.2).
                                           This will succeed if the specified treasure is within the minimum sample distance.

            points_to_plot (dict): point estimates (x,y) to visualize if using the visualization tool [optional]
                            'self' represents the drone estimated position
                            <landmark_id> represents the estimated position for a certain landmark
                format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        # Update SLAM with the last movement we commanded (if any)
        if self.last_distance > 0 or self.last_steering != 0:
            self.slam.process_movement(self.last_distance, self.last_steering)
            # Reset for next iteration
            self.last_distance = 0.0
            self.last_steering = 0.0
        
        # Use SLAM to process measurements and get both drone and landmark estimates
        self.slam.process_measurements(measurements)
        
        # Get all coordinates from SLAM
        slam_coordinates = self.slam.get_coordinates()
        self.drone_x = slam_coordinates['self'][0]
        self.drone_y = slam_coordinates['self'][1]
        self.landmarks = {k: v for k, v in slam_coordinates.items() if k != 'self'}
        
        # Check if we're close enough to extract treasure
        treasure_x = treasure_location['x']
        treasure_y = treasure_location['y']
        distance_to_treasure = math.sqrt((self.drone_x - treasure_x)**2 + (self.drone_y - treasure_y)**2)
        
        if distance_to_treasure <= 0.25:
            # Extract treasure
            action = f"extract * {self.drone_x:.1f} {self.drone_y:.1f}"
            points_to_plot = self._get_points_to_plot()
            return action, points_to_plot
        
        # Simple greedy approach: move towards treasure while avoiding obstacles
        action = self._get_safe_move_towards_treasure(treasure_x, treasure_y)
        
        # Store the movement we're about to command so we can update SLAM next time
        if action.startswith('move'):
            parts = action.split()
            self.last_distance = float(parts[1])
            self.last_steering = float(parts[2])
        
        points_to_plot = self._get_points_to_plot()
        return action, points_to_plot
    
    def _get_safe_move_towards_treasure(self, treasure_x: float, treasure_y: float):
        """Get a safe move towards the treasure."""
        # Calculate desired direction to treasure
        dx = treasure_x - self.drone_x
        dy = treasure_y - self.drone_y
        desired_bearing = math.atan2(dy, dx)
        
        # Get current bearing from SLAM
        current_bearing = self.slam.drone_bearing
        
        # Check if direct path to treasure is safe
        if self._is_path_safe_to_point(treasure_x, treasure_y):
            # Direct path is safe, move towards treasure
            steering = desired_bearing - current_bearing
            steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
            max_safe_steering = self.max_steering * 0.7  # Reduced from 0.9 for smaller turns
            steering = max(-max_safe_steering, min(max_safe_steering, steering))
            
            distance = math.sqrt(dx**2 + dy**2)
            move_distance = min(distance, self.max_distance * 0.3)  # Reduced from 0.5 for smaller steps
            if move_distance < 0.1:
                move_distance = 0.1
        else:
            # Direct path is blocked, find safe direction
            move_distance, steering = self._find_safe_direction(treasure_x, treasure_y, current_bearing)
        
        return f"move {move_distance:.3f} {steering:.3f}"
    
    def _is_path_safe_to_point(self, target_x: float, target_y: float):
        """Check if direct path to a point is safe (no tree collisions)."""
        for landmark_id, (tree_x, tree_y) in self.landmarks.items():
            # Get tree radius from measurements (approximate)
            tree_radius = 0.5  # Default radius
            
            # Add safety buffer around trees (extra caution)
            safety_buffer = 0.30  # 15cm safety buffer (reduced from 30cm)
            effective_radius = tree_radius + safety_buffer
            
            if self._line_circle_intersect((self.drone_x, self.drone_y), (target_x, target_y), (tree_x, tree_y), effective_radius):
                return False
        return True
    
    def _line_circle_intersect(self, first_point, second_point, origin, radius):
        """Check if a line segment intersects a circle."""
        x1, y1 = first_point
        x2, y2 = second_point
        ox, oy = origin
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
            return c <= 0
        else:
            if disc <= 0:
                return False
            sqrtdisc = math.sqrt(disc)
            t1 = (-b + sqrtdisc)/(2*a)
            t2 = (-b - sqrtdisc)/(2*a)
            return (0 < t1 and t1 < 1) or (0 < t2 and t2 < 1)
    
    def _find_safe_direction(self, treasure_x: float, treasure_y: float, current_bearing: float):
        """Find a safe direction to move when direct path is blocked."""
        # Try different angles around the desired direction
        desired_bearing = math.atan2(treasure_y - self.drone_y, treasure_x - self.drone_x)
        
        # Try smaller angle increments to avoid large turns
        for angle_offset in [0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0]:
            test_bearing = desired_bearing + angle_offset
            test_x = self.drone_x + self.max_distance * 0.25 * math.cos(test_bearing)  # Smaller test distance
            test_y = self.drone_y + self.max_distance * 0.25 * math.sin(test_bearing)
            
            if self._is_path_safe_to_point(test_x, test_y):
                # Found safe direction
                steering = test_bearing - current_bearing
                steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
                max_safe_steering = self.max_steering * 0.7  # Reduced for smaller turns
                steering = max(-max_safe_steering, min(max_safe_steering, steering))
                
                return self.max_distance * 0.25, steering  # Smaller movement distance
        
        # If no safe direction found, try moving perpendicular to obstacles
        # This is a fallback strategy
        steering = 0.3  # Reduced from 0.5 for smaller turn
        max_safe_steering = self.max_steering * 0.7
        steering = max(-max_safe_steering, min(max_safe_steering, steering))
        
        return self.max_distance * 0.15, steering  # Even smaller distance for safety
    
    def _get_points_to_plot(self):
        """Get points to plot for visualization."""
        points = {'self': (self.drone_x, self.drone_y)}
        points.update(self.landmarks)
        return points

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
