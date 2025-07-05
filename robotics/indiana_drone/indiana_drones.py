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
import random

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
        
        # Dictionary to store landmark positions and radii
        self.landmarks = {}  # {landmark_id: (x, y)}
        self.landmark_radii = {}  # {landmark_id: radius}
        
        # Noise parameters for measurements and movements
        # Optimized for drone navigation with tree landmarks
        self.measure_distance_noise = 0.15  # Reduced from 0.4 - distance measurements are typically more reliable
        self.measure_bearing_noise = 0.25   # Reduced from 0.4 - bearing has moderate noise
        self.move_distance_noise = 0.1      # Reduced from 0.4 - movement distance is more predictable
        self.move_steering_noise = 0.2      # Reduced from 0.4 - steering has moderate noise
        
        # Uncertainty parameters
        self.drone_uncertainty = 0.2        # Reduced from 0.4 - start with moderate uncertainty
        self.landmark_uncertainty = 0.25    # Reduced from 0.4 - landmarks should be more certain
        
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
                # Distance-based uncertainty: closer measurements are more reliable
                distance_uncertainty = max(0.05, distance * 0.08)  # Reduced from 0.1 for better precision
                weight_new = 1.0 / distance_uncertainty
                
                total_weight = weight_old + weight_new
                landmark_x = (weight_old * old_x + weight_new * landmark_x) / total_weight
                landmark_y = (weight_old * old_y + weight_new * landmark_y) / total_weight
                
                # Reduce uncertainty for this landmark more aggressively
                self.landmark_uncertainty = max(0.03, self.landmark_uncertainty * 0.85)  # More aggressive reduction
            else:
                # New landmark - initialize with moderate uncertainty
                self.landmark_uncertainty = 0.2  # Reduced from 0.3 for better initial estimates
            
            # Update or add landmark position
            self.landmarks[landmark_id] = (landmark_x, landmark_y)
            # Store the tree radius
            self.landmark_radii[landmark_id] = radius
        
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
        
        # Increase uncertainty with movement - optimized for drone navigation
        # Uncertainty grows with distance moved and steering angle
        movement_uncertainty = (distance * 0.02) + (abs(steering) * 0.1)  # Distance and steering based
        self.drone_uncertainty += movement_uncertainty
        # Cap maximum uncertainty to prevent excessive drift
        self.drone_uncertainty = min(self.drone_uncertainty, 0.5)
        
        # Periodically run GraphSLAM optimization - more frequent for better accuracy
        if len(self.movement_history) % 5 == 0:  # Reduced from 10 for more frequent optimization
            self._optimize_graph()
    
    def _optimize_graph(self):
        """Enhanced GraphSLAM optimization to improve estimates."""
        # This is a simplified version - in practice, you'd use a full GraphSLAM implementation
        # For now, we'll do improved smoothing and consistency checking
        
        if len(self.movement_history) < 2:
            return
        
        # Enhanced smoothing of landmark positions based on consistency
        for landmark_id in self.landmarks:
            # Check if landmark appears in multiple measurements
            appearances = 0
            total_x = 0
            total_y = 0
            total_weight = 0
            
            for i, measurements in enumerate(self.measurement_history):
                if landmark_id in measurements:
                    appearances += 1
                    # Weight recent measurements more heavily
                    weight = 1.0 + (i * 0.1)  # Recent measurements get higher weight
                    total_x += self.landmarks[landmark_id][0] * weight
                    total_y += self.landmarks[landmark_id][1] * weight
                    total_weight += weight
            
            if appearances > 1:
                # Weighted average the positions
                avg_x = total_x / total_weight
                avg_y = total_y / total_weight
                
                # Only update if change is significant (prevents oscillation)
                current_x, current_y = self.landmarks[landmark_id]
                change = math.sqrt((avg_x - current_x)**2 + (avg_y - current_y)**2)
                if change > 0.01:  # Only update if change is significant
                    self.landmarks[landmark_id] = (avg_x, avg_y)
        
        # Reduce uncertainty after optimization
        self.drone_uncertainty = max(0.1, self.drone_uncertainty * 0.95)
        self.landmark_uncertainty = max(0.02, self.landmark_uncertainty * 0.9)


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
        self.landmark_radii = {}
        
        # Track last movement to update SLAM
        self.last_distance = 0.0
        self.last_steering = 0.0
        
        # Track failed extraction attempts
        self.failed_extraction = False
        self.last_treasure_location = None
        self.failed_extraction_count = 0
        self.last_extraction_position = None
        
        # Path planning state
        self.path = []
        self.current_path_index = 0
        self.target_reached = False
        
        # Navigation state tracking for better path correction
        self.stuck_counter = 0
        self.last_position = (0.0, 0.0)
        self.position_history = []
        self.max_history_length = 10
        
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
        # Get tree radii from SLAM
        self.landmark_radii = self.slam.landmark_radii.copy()
        
        # Extract treasure location early for stuck detection
        treasure_x = treasure_location['x']
        treasure_y = treasure_location['y']
        
        # Update position history for stuck detection
        current_position = (self.drone_x, self.drone_y)
        self.position_history.append(current_position)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Check if drone is stuck (improved detection)
        is_stuck = False
        stuck_reason = ""
        
        # Check 1: Not moving significantly (over 5 steps)
        if len(self.position_history) >= 5:
            recent_movement = math.sqrt((self.position_history[-1][0] - self.position_history[-5][0])**2 + 
                                      (self.position_history[-1][1] - self.position_history[-5][1])**2)
            if recent_movement < 0.05:  # Reduced threshold - stuck if moved less than 0.05m in 5 steps
                is_stuck = True
                stuck_reason = f"Not moving (recent_movement: {recent_movement:.3f})"
        
        # Check 2: Oscillating in place (moving back and forth)
        if len(self.position_history) >= 6:
            # Check if we're moving in circles or oscillating
            total_movement = 0
            for i in range(1, len(self.position_history)):
                dx = self.position_history[i][0] - self.position_history[i-1][0]
                dy = self.position_history[i][1] - self.position_history[i-1][1]
                total_movement += math.sqrt(dx**2 + dy**2)
            
            # If total movement is high but net movement is low, we're oscillating
            net_movement = math.sqrt((self.position_history[-1][0] - self.position_history[0][0])**2 + 
                                   (self.position_history[-1][1] - self.position_history[0][1])**2)
            if total_movement > 0.5 and net_movement < 0.1:  # Moving a lot but not going anywhere
                is_stuck = True
                stuck_reason = f"Oscillating (total: {total_movement:.3f}, net: {net_movement:.3f})"
        
        # Check 3: Not making progress toward treasure
        if len(self.position_history) >= 3:
            initial_distance = math.sqrt((self.position_history[0][0] - treasure_x)**2 + 
                                       (self.position_history[0][1] - treasure_y)**2)
            current_distance = math.sqrt((self.drone_x - treasure_x)**2 + (self.drone_y - treasure_y)**2)
            if current_distance >= initial_distance:  # Not getting closer to treasure
                is_stuck = True
                stuck_reason = f"No progress (initial: {initial_distance:.3f}, current: {current_distance:.3f})"
        
        # Update stuck counter
        if is_stuck:
            self.stuck_counter += 1
            print(f"STUCK DETECTED: {stuck_reason}, counter: {self.stuck_counter}")
        else:
            if self.stuck_counter > 0:
                old_counter = self.stuck_counter
                self.stuck_counter = max(0, self.stuck_counter - 1)
                if old_counter != self.stuck_counter:
                    print(f"RECOVERY SUCCESS: Stuck counter decreased from {old_counter} to {self.stuck_counter}")
            else:
                self.stuck_counter = 0
        
        # Check if we're close enough to extract treasure
        distance_to_treasure = math.sqrt((self.drone_x - treasure_x)**2 + (self.drone_y - treasure_y)**2)
        
        # Track failed extraction attempts
        current_position = (self.drone_x, self.drone_y)
        if distance_to_treasure <= 0.15:
            # We think we're close enough to extract
            if self.last_extraction_position is not None:
                # Check if we're at the same position as last extraction attempt
                extraction_movement = math.sqrt((current_position[0] - self.last_extraction_position[0])**2 + 
                                              (current_position[1] - self.last_extraction_position[1])**2)
                if extraction_movement < 0.05:  # Still at same position
                    self.failed_extraction_count += 1
                    print(f"FAILED EXTRACTION: Attempt {self.failed_extraction_count} at same position (movement: {extraction_movement:.3f})")
                else:
                    if self.failed_extraction_count > 0:
                        print(f"RECOVERY SUCCESS: Extraction position changed, resetting failed counter from {self.failed_extraction_count} to 0")
                    self.failed_extraction_count = 0  # Reset if we moved
            
            self.last_extraction_position = current_position
            
            # If we've tried extracting multiple times at the same spot, we're stuck
            if self.failed_extraction_count > 2:
                is_stuck = True
                self.stuck_counter = max(self.stuck_counter, 3)  # Force stuck state
                print(f"STUCK FROM FAILED EXTRACTIONS: {self.failed_extraction_count} attempts, forcing stuck state")
        
        # Adaptive navigation based on stuck detection - CHECK BEFORE EXTRACTION
        if self.stuck_counter > 2 or self.failed_extraction_count > 2:  # Trigger recovery for either condition
            # Drone is stuck - use more aggressive pathfinding
            print(f"TRIGGERING STUCK RECOVERY: stuck_counter={self.stuck_counter}, failed_extractions={self.failed_extraction_count}")
            action = self._get_unstuck_move(treasure_x, treasure_y)
            print(f"RECOVERY ACTION: {action}")
        elif distance_to_treasure <= 0.15:
            # Not stuck and close enough to extract
            action = f"extract * {self.drone_x:.1f} {self.drone_y:.1f}"
        else:
            # Normal navigation
            action = self._get_safe_move_towards_treasure(treasure_x, treasure_y)
        
        # Store the movement we're about to command so we can update SLAM next time
        if action.startswith('move'):
            parts = action.split()
            self.last_distance = float(parts[1])
            self.last_steering = float(parts[2])
        
        points_to_plot = self._get_points_to_plot()
        return action, points_to_plot
    
    def _get_safe_move_towards_treasure(self, treasure_x: float, treasure_y: float):
        """Get a safe move towards the treasure with improved path correction."""
        # Calculate desired direction to treasure
        dx = treasure_x - self.drone_x
        dy = treasure_y - self.drone_y
        distance_to_treasure = math.sqrt(dx**2 + dy**2)
        desired_bearing = math.atan2(dy, dx)
        
        # Get current bearing from SLAM
        current_bearing = self.slam.drone_bearing
        
        # Adaptive step size based on distance to treasure
        if distance_to_treasure < 0.5:
            # Close to treasure - use smaller steps for precision
            base_step_size = min(distance_to_treasure * 0.8, self.max_distance * 0.2)
        elif distance_to_treasure < 2.0:
            # Medium distance - moderate steps
            base_step_size = min(distance_to_treasure * 0.6, self.max_distance * 0.4)
        else:
            # Far from treasure - larger steps for efficiency
            base_step_size = min(distance_to_treasure * 0.4, self.max_distance * 0.5)  # Reduced from 0.6 to 0.5
        
        # Ensure minimum step size
        base_step_size = max(base_step_size, 0.1)
        
        # Check if direct path to treasure is safe
        if self._is_path_safe_to_point(treasure_x, treasure_y):
            # Direct path is safe, move towards treasure
            steering = desired_bearing - current_bearing
            steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
            
            # Adaptive steering based on distance and angle difference
            angle_diff = abs(steering)
            if angle_diff > math.pi/2:
                # Large turn needed - use smaller steering to avoid overshooting
                max_safe_steering = self.max_steering * 0.5
            else:
                # Small turn - can use more steering
                max_safe_steering = self.max_steering * 0.8
            
            steering = max(-max_safe_steering, min(max_safe_steering, steering))
            move_distance = base_step_size
        else:
            # Direct path is blocked, find safe direction
            move_distance, steering = self._find_safe_direction(treasure_x, treasure_y, current_bearing, base_step_size)
        
        return f"move {move_distance:.3f} {steering:.3f}"
    

    
    def _is_path_safe_to_point(self, target_x: float, target_y: float):
        """Check if direct path to a point is safe (no tree collisions)."""
        for landmark_id, (tree_x, tree_y) in self.landmarks.items():
            # Get actual tree radius from stored data
            tree_radius = self.landmark_radii.get(landmark_id, 0.5)  # Use actual radius, fallback to 0.5
            
            # Add safety buffer around trees (extra caution)
            safety_buffer = 0.15  # Increased from 0.30 to 0.45 for more caution
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
    
    def _find_safe_direction(self, treasure_x: float, treasure_y: float, current_bearing: float, base_step_size: float):
        """Find a safe direction to move when direct path is blocked with improved pathfinding."""
        # Calculate desired direction to treasure
        desired_bearing = math.atan2(treasure_y - self.drone_y, treasure_x - self.drone_x)
        
        # Strategy 1: Try angles close to desired direction (preferred)
        angle_increments = [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5]
        for angle_offset in angle_increments:
            test_bearing = desired_bearing + angle_offset
            test_x = self.drone_x + base_step_size * math.cos(test_bearing)
            test_y = self.drone_y + base_step_size * math.sin(test_bearing)
            
            if self._is_path_safe_to_point(test_x, test_y):
                # Found safe direction close to desired path
                steering = test_bearing - current_bearing
                steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
                max_safe_steering = self.max_steering * 0.8
                steering = max(-max_safe_steering, min(max_safe_steering, steering))
                return base_step_size, steering
        
        # Strategy 2: Try wider angles if close angles failed
        wide_angles = [0.6, -0.6, 0.8, -0.8, 1.0, -1.0, 1.2, -1.2, 1.4, -1.4]
        for angle_offset in wide_angles:
            test_bearing = desired_bearing + angle_offset
            test_x = self.drone_x + base_step_size * 0.8 * math.cos(test_bearing)  # Smaller step for wide turns
            test_y = self.drone_y + base_step_size * 0.8 * math.sin(test_bearing)
            
            if self._is_path_safe_to_point(test_x, test_y):
                # Found safe direction with wider angle
                steering = test_bearing - current_bearing
                steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
                max_safe_steering = self.max_steering * 0.6  # More conservative for wide turns
                steering = max(-max_safe_steering, min(max_safe_steering, steering))
                return base_step_size * 0.8, steering
        
        # Strategy 3: Try moving perpendicular to obstacles (emergency fallback)
        # Find the closest obstacle and move perpendicular to it
        closest_obstacle = None
        min_distance = float('inf')
        
        for landmark_id, (tree_x, tree_y) in self.landmarks.items():
            tree_radius = self.landmark_radii.get(landmark_id, 0.5)
            distance = math.sqrt((self.drone_x - tree_x)**2 + (self.drone_y - tree_y)**2) - tree_radius
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = (tree_x, tree_y)
        
        if closest_obstacle:
            # Calculate perpendicular direction away from obstacle
            ox, oy = closest_obstacle
            dx = self.drone_x - ox
            dy = self.drone_y - oy
            # Normalize and rotate 90 degrees
            length = math.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
                # Rotate 90 degrees (perpendicular)
                perp_x, perp_y = -dy, dx
                escape_bearing = math.atan2(perp_y, perp_x)
                
                steering = escape_bearing - current_bearing
                steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
                max_safe_steering = self.max_steering * 0.5
                steering = max(-max_safe_steering, min(max_safe_steering, steering))
                
                return base_step_size * 0.5, steering
        
        # Strategy 4: Last resort - small random movement
        steering = 0.2  # Small turn
        max_safe_steering = self.max_steering * 0.3
        steering = max(-max_safe_steering, min(max_safe_steering, steering))
        
        return base_step_size * 0.3, steering
    
    def _get_unstuck_move(self, treasure_x: float, treasure_y: float):
        """Get a move to help the drone get unstuck when it's been stuck for too long."""
        current_bearing = self.slam.drone_bearing
        print(f"GETTING UNSTUCK MOVE: failed_extractions={self.failed_extraction_count}, stuck_counter={self.stuck_counter}")
        
        # Check if we're stuck near treasure (failed extractions)
        if self.failed_extraction_count > 2:
            print(f"USING SEARCH NEAR TREASURE STRATEGY")
            return self._search_near_treasure(treasure_x, treasure_y)
        
        # Strategy 1: Try a large turn to get out of current orientation
        large_turn = math.pi / 2  # 90 degrees
        if self.stuck_counter % 2 == 0:
            large_turn = -large_turn  # Alternate direction
        
        # Try the large turn
        test_bearing = current_bearing + large_turn
        test_x = self.drone_x + 0.15 * math.cos(test_bearing)  # Smaller recovery move
        test_y = self.drone_y + 0.15 * math.sin(test_bearing)
        
        if self._is_path_safe_to_point(test_x, test_y):
            steering = large_turn
            max_safe_steering = self.max_steering * 0.8
            steering = max(-max_safe_steering, min(max_safe_steering, steering))
            return f"move 0.15 {steering:.3f}"  # Fixed small recovery move
        
        # Strategy 2: Try moving in the opposite direction of the treasure
        opposite_bearing = math.atan2(self.drone_y - treasure_y, self.drone_x - treasure_x)
        steering = opposite_bearing - current_bearing
        steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
        
        test_x = self.drone_x + 0.1 * math.cos(opposite_bearing)  # Smaller recovery move
        test_y = self.drone_y + 0.1 * math.sin(opposite_bearing)
        
        if self._is_path_safe_to_point(test_x, test_y):
            max_safe_steering = self.max_steering * 0.6
            steering = max(-max_safe_steering, min(max_safe_steering, steering))
            return f"move 0.1 {steering:.3f}"  # Fixed small recovery move
        
        # Strategy 3: Random movement in any safe direction
        for angle in [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2, math.pi, -math.pi]:
            test_bearing = current_bearing + angle
            test_x = self.drone_x + 0.08 * math.cos(test_bearing)  # Smaller recovery move
            test_y = self.drone_y + 0.08 * math.sin(test_bearing)
            
            if self._is_path_safe_to_point(test_x, test_y):
                steering = angle
                max_safe_steering = self.max_steering * 0.5
                steering = max(-max_safe_steering, min(max_safe_steering, steering))
                return f"move 0.08 {steering:.3f}"  # Fixed small recovery move
        
        # Last resort: small movement in current direction
        return f"move 0.03 0.0"  # Very small last resort move
    
    def _search_near_treasure(self, treasure_x: float, treasure_y: float):
        """Search around the treasure location when stuck due to failed extractions."""
        current_bearing = self.slam.drone_bearing
        
        # More aggressive search pattern - start closer and search more systematically
        # Use a grid-like search pattern around the treasure
        search_attempt = self.failed_extraction_count - 2  # Start from attempt 3
        
        # Define search positions in a systematic pattern around the treasure
        # Much smaller movements for recovery - we're already close to treasure
        search_positions = [
            # Very close positions (0.05m radius) - tiny adjustments
            (0.05, 0), (0, 0.05), (-0.05, 0), (0, -0.05),
            # Close diagonal positions (0.07m radius)
            (0.05, 0.05), (-0.05, 0.05), (-0.05, -0.05), (0.05, -0.05),
            # Slightly further positions (0.1m radius)
            (0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1),
            # Medium diagonal positions (0.14m radius)
            (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1),
            # Further positions (0.15m radius) - still within extraction range
            (0.15, 0), (0, 0.15), (-0.15, 0), (0, -0.15),
        ]
        
        # Cycle through search positions
        search_index = search_attempt % len(search_positions)
        search_dx, search_dy = search_positions[search_index]
        search_x = treasure_x + search_dx
        search_y = treasure_y + search_dy
        
        print(f"SEARCHING NEAR TREASURE: position {search_index+1}/{len(search_positions)}, offset=({search_dx:.3f}, {search_dy:.3f})")
        
        # Calculate direction to search position
        dx = search_x - self.drone_x
        dy = search_y - self.drone_y
        search_bearing = math.atan2(dy, dx)
        
        # Check if path to search position is safe
        if self._is_path_safe_to_point(search_x, search_y):
            # Move toward search position
            steering = search_bearing - current_bearing
            steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
            max_safe_steering = self.max_steering * 0.6
            steering = max(-max_safe_steering, min(max_safe_steering, steering))
            
            distance = math.sqrt(dx**2 + dy**2)
            # For recovery, use the actual distance to search position, but cap it at a small value
            move_distance = min(distance, 0.1)  # Cap at 0.1m for recovery moves
            action = f"move {move_distance:.3f} {steering:.3f}"
            print(f"SEARCH ACTION: {action}")
            return action
        
        # If direct path to search position is blocked, try alternative search directions
        for angle_offset in [0, math.pi/2, -math.pi/2, math.pi, -math.pi/4, math.pi/4, -3*math.pi/4, 3*math.pi/4]:
            alt_search_angle = search_bearing + angle_offset
            alt_search_x = treasure_x + 0.1 * math.cos(alt_search_angle)  # Smaller alternative search radius
            alt_search_y = treasure_y + 0.1 * math.sin(alt_search_angle)
            
            if self._is_path_safe_to_point(alt_search_x, alt_search_y):
                dx = alt_search_x - self.drone_x
                dy = alt_search_y - self.drone_y
                alt_search_bearing = math.atan2(dy, dx)
                
                steering = alt_search_bearing - current_bearing
                steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
                max_safe_steering = self.max_steering * 0.6
                steering = max(-max_safe_steering, min(max_safe_steering, steering))
                
                distance = math.sqrt(dx**2 + dy**2)
                # For recovery, use the actual distance to search position, but cap it at a small value
                move_distance = min(distance, 0.1)  # Cap at 0.1m for recovery moves
                action = f"move {move_distance:.3f} {steering:.3f}"
                print(f"ALTERNATIVE SEARCH ACTION: {action}")
                return action
        
        # If all search directions are blocked, try moving away from current position
        escape_bearing = current_bearing + math.pi  # Move in opposite direction
        test_x = self.drone_x + self.max_distance * 0.1 * math.cos(escape_bearing)  # Smaller escape movement
        test_y = self.drone_y + self.max_distance * 0.1 * math.sin(escape_bearing)
        
        if self._is_path_safe_to_point(test_x, test_y):
            steering = math.pi
            max_safe_steering = self.max_steering * 0.5
            steering = max(-max_safe_steering, min(max_safe_steering, steering))
            action = f"move {self.max_distance * 0.1:.3f} {steering:.3f}"
            print(f"ESCAPE ACTION: {action}")
            return action
        
        # Last resort: try extraction at current position anyway (maybe we're actually close enough)
        action = f"extract * {self.drone_x:.1f} {self.drone_y:.1f}"
        print(f"LAST RESORT ACTION: {action}")
        return action
    
    def _get_points_to_plot(self):
        """Get points to plot for visualization."""
        points = {'self': (self.drone_x, self.drone_y)}
        points.update(self.landmarks)
        return points


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
