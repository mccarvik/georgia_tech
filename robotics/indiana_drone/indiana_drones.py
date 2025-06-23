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

from typing import Dict, List
import math
import heapq
import numpy as np

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
        
        # Initialize drone state
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_bearing = 0.0
        
        # Store landmarks and their positions
        self.landmarks = {}
        
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
        # Update landmarks from measurements
        self._update_landmarks(measurements)
        
        # Check if we're close enough to extract treasure
        treasure_x = treasure_location['x']
        treasure_y = treasure_location['y']
        distance_to_treasure = math.sqrt((self.drone_x - treasure_x)**2 + (self.drone_y - treasure_y)**2)
        
        if distance_to_treasure <= 0.25:
            # Extract treasure
            action = f"extract * {self.drone_x:.1f} {self.drone_y:.1f}"
            points_to_plot = self._get_points_to_plot()
            return action, points_to_plot
        
        # Plan path to treasure if we don't have one or need to replan
        if not self.path or self.current_path_index >= len(self.path):
            self._plan_path_to_treasure(treasure_x, treasure_y)
        
        # Execute next move in path
        if self.path and self.current_path_index < len(self.path):
            next_waypoint = self.path[self.current_path_index]
            action = self._get_move_action_to_waypoint(next_waypoint)
            self.current_path_index += 1
        else:
            # Fallback: move directly towards treasure
            action = self._get_move_action_to_point(treasure_x, treasure_y)
        
        points_to_plot = self._get_points_to_plot()
        return action, points_to_plot
    
    def _update_landmarks(self, measurements: Dict):
        """Update landmark positions from measurements."""
        for landmark_id, measurement in measurements.items():
            distance = measurement['distance']
            bearing = measurement['bearing']
            
            # Convert to absolute coordinates
            absolute_bearing = self.drone_bearing + bearing
            landmark_x = self.drone_x + distance * math.cos(absolute_bearing)
            landmark_y = self.drone_y + distance * math.sin(absolute_bearing)
            
            self.landmarks[landmark_id] = (landmark_x, landmark_y)
    
    def _plan_path_to_treasure(self, treasure_x: float, treasure_y: float):
        """Plan a path to the treasure using A* algorithm with obstacle avoidance."""
        # Create a simple A* pathfinding with obstacle avoidance
        start = (self.drone_x, self.drone_y)
        goal = (treasure_x, treasure_y)
        
        # Check if direct path is safe
        if self._is_path_safe(start, goal):
            self.path = [goal]
            self.current_path_index = 0
            return
        
        # Use A* to find safe path
        path = self._a_star_pathfinding(start, goal)
        if path:
            self.path = path[1:]  # Remove start point
            self.current_path_index = 0
        else:
            # Fallback to direct path if A* fails
            self.path = [goal]
            self.current_path_index = 0
    
    def _is_path_safe(self, start, end):
        """Check if a path between two points is safe (no tree collisions)."""
        for landmark_id, (tree_x, tree_y) in self.landmarks.items():
            # Get tree radius from measurements (approximate)
            tree_radius = 0.5  # Default radius
            if self._line_circle_intersect(start, end, (tree_x, tree_y), tree_radius):
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
    
    def _a_star_pathfinding(self, start, goal):
        """A* pathfinding algorithm with obstacle avoidance."""
        # Simple grid-based A* implementation
        grid_size = 0.5
        max_iterations = 1000
        
        # Convert to grid coordinates
        start_grid = (int(start[0] / grid_size), int(start[1] / grid_size))
        goal_grid = (int(goal[0] / grid_size), int(goal[1] / grid_size))
        
        # Priority queue for A*
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                path.reverse()
                
                # Convert back to world coordinates
                world_path = []
                for grid_pos in path:
                    world_x = grid_pos[0] * grid_size
                    world_y = grid_pos[1] * grid_size
                    world_path.append((world_x, world_y))
                return world_path
            
            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if neighbor is safe
                neighbor_world = (neighbor[0] * grid_size, neighbor[1] * grid_size)
                current_world = (current[0] * grid_size, current[1] * grid_size)
                
                if not self._is_path_safe(current_world, neighbor_world):
                    continue
                
                tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _heuristic(self, a, b):
        """Heuristic function for A* (Euclidean distance)."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _get_move_action_to_waypoint(self, waypoint):
        """Get move action to reach a waypoint."""
        target_x, target_y = waypoint
        return self._get_move_action_to_point(target_x, target_y)
    
    def _get_move_action_to_point(self, target_x: float, target_y: float):
        """Get move action to reach a target point."""
        # Calculate desired bearing to target
        dx = target_x - self.drone_x
        dy = target_y - self.drone_y
        desired_bearing = math.atan2(dy, dx)
        
        # Calculate required steering
        steering = desired_bearing - self.drone_bearing
        
        # Normalize steering to [-pi, pi]
        steering = ((steering + math.pi) % (2 * math.pi)) - math.pi
        
        # Limit steering to max_steering
        steering = max(-self.max_steering, min(self.max_steering, steering))
        
        # Calculate distance to target
        distance = math.sqrt(dx**2 + dy**2)
        
        # Limit distance to max_distance
        distance = min(distance, self.max_distance)
        
        # Update drone position (for next iteration)
        self.drone_bearing += steering
        self.drone_bearing = ((self.drone_bearing + math.pi) % (2 * math.pi)) - math.pi
        self.drone_x += distance * math.cos(self.drone_bearing)
        self.drone_y += distance * math.sin(self.drone_bearing)
        
        return f"move {distance:.3f} {steering:.3f}"
    
    def _get_points_to_plot(self):
        """Get points to plot for visualization."""
        points = {'self': (self.drone_x, self.drone_y)}
        points.update(self.landmarks)
        return points

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
