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
import heapq
import pdb

# If you see different scores locally and on Gradescope this may be an indication
# that you are uploading a different file than the one you are executing locally.
# If this local ID doesn't match the ID on Gradescope then you uploaded a different file.
OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib

    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')


class DeliveryPlanner_PartA:
    """
    Note: All print outs must be conditioned on the debug parameter.

    Required methods in this class are:

      plan_delivery(self, debug = False):
       Stubbed out below.  You may not change the method signature
        as it will be called directly by the autograder but you
        may modify the internals as needed.

      __init__:
        Required to initialize the class.  Signature can NOT be changed.
        Basic template starter code is provided.  You may choose to
        use this starter code or modify and replace it based on
        your own solution.
    """

    def __init__(self, warehouse_viewer, dropzone_location, todo, box_locations):
        self.warehouse_viewer = warehouse_viewer
        self.dropzone_location = dropzone_location
        self.todo = todo
        self.box_locations = box_locations
        
        # You may use these symbols indicating direction for visual debugging
        # ['^', '<', 'v', '>', '\\', '/', '[', ']']
        # or you may choose to use arrows instead
        # ['🡑', '🡐', '🡓', '🡒',  '🡔', '🡕', '🡖', '🡗']
        
        # Define movement costs and directions
        self.movement_costs = {
            'n': 2, 's': 2, 'e': 2, 'w': 2,  # Cardinal directions
            'ne': 3, 'nw': 3, 'se': 3, 'sw': 3  # Diagonal directions
        }
        
        # Define direction vectors
        self.directions = {
            'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1),
            'ne': (-1, 1), 'nw': (-1, -1), 'se': (1, 1), 'sw': (1, -1)
        }
        
        # Track delivered boxes
        self.delivered_boxes = set()

    def _is_valid_move(self, pos):
        """Check if a position is valid (within bounds and not a wall)."""
        # Check if position is within grid bounds
        if pos[0] < 0 or pos[0] >= len(self.warehouse_viewer) or \
           pos[1] < 0 or pos[1] >= len(self.warehouse_viewer[0]):
            return False
            
        # Check if position is not a wall
        return self.warehouse_viewer[pos[0]][pos[1]] != '#'


    def _get_neighbors(self, pos, carrying_box=None, cell_cache=None):
        """Get valid neighboring positions with their costs."""
        neighbors = []
        for direction, (dx, dy) in self.directions.items():
            new_pos = (pos[0] + dx, pos[1] + dy)
            
            # Check if we've already cached this cell
            if cell_cache is not None and new_pos in cell_cache:
                cell = cell_cache[new_pos]
            else:
                try:
                    cell = self.warehouse_viewer[new_pos[0]][new_pos[1]]
                    if cell_cache is not None:
                        cell_cache[new_pos] = cell
                except IndexError:
                    continue  # Skip if position is out of bounds
            
            if self._is_valid_move_cell(cell, carrying_box):
                cost = self.movement_costs[direction]
                neighbors.append((new_pos, direction, cost))
        return neighbors


    def _heuristic(self, pos, target):
        """Heuristic for getting adjacent to target (box or dropzone)."""
        dx = abs(pos[0] - target[0])
        dy = abs(pos[1] - target[1])
        
        # If we're already adjacent, heuristic is 0
        if dx <= 1 and dy <= 1 and (dx + dy) > 0:
            print(f"  Heuristic: pos={pos}, target={target} - ALREADY ADJACENT, cost=0")
            return 0
        
        # Calculate minimum moves to get adjacent
        # We need to get within 1 step in both x and y directions
        moves_x = max(0, dx - 1)  # How many x moves to get within 1 step
        moves_y = max(0, dy - 1)  # How many y moves to get within 1 step
        
        # Use diagonal movement when possible to minimize total moves
        diagonal_moves = min(moves_x, moves_y)
        straight_moves = max(moves_x, moves_y) - diagonal_moves
        
        # Calculate cost using actual movement costs
        heuristic_cost = diagonal_moves * 3 + straight_moves * 2
        
        print(f"  Heuristic: pos={pos}, target={target}, dx={dx}, dy={dy}, moves_x={moves_x}, moves_y={moves_y}, diagonal={diagonal_moves}, straight={straight_moves}, cost={heuristic_cost}")
        
        return heuristic_cost


    def _a_star_search(self, start, goal, carrying_box=None):
        """A* search algorithm to find optimal path to get adjacent to goal."""
        
        # Initialize frontier as a priority queue
        frontier = []
        heapq.heappush(frontier, (0, start, []))  # (f_cost, position, path)
        
        # Track visited positions and their costs
        visited = {start: 0}  # position -> g_cost
        
        # Cache for cell contents to avoid repeated viewing
        cell_cache = {start: self.warehouse_viewer[start[0]][start[1]]}
        
        if carrying_box is None:
            print(f"\nA* Search: {start} -> adjacent to box at {goal}")
        else:
            print(f"\nA* Search: {start} -> adjacent to dropzone at {goal}")
        
        iteration = 0
        while frontier:
            iteration += 1
            f_cost, pos, path = heapq.heappop(frontier)
            g_cost = visited[pos]
            h_cost = self._heuristic(pos, goal)
            
            print(f"Iteration {iteration}: pos={pos}, g_cost={g_cost}, h_cost={h_cost}, f_cost={f_cost}, path={path}")
            
            # Check if we've reached the goal position directly
            if pos == goal:
                print(f"Reached goal directly at {pos}")
                return path
            
            # Check if we're adjacent to the goal (this is what we really want)
            if not carrying_box and self._is_adjacent_to_box(pos, goal):
                print(f"Reached adjacent to box at {pos}, box at {goal}")
                return path
            elif carrying_box and self._is_adjacent_to_dropzone(pos, goal):
                print(f"Reached adjacent to dropzone at {pos}, dropzone at {goal}")
                return path
                
            # Explore all neighbors
            for next_pos, direction, move_cost in self._get_neighbors(pos, carrying_box, cell_cache):
                new_g_cost = g_cost + move_cost
                
                # Only explore if we found a better path or haven't seen this position
                if next_pos not in visited or new_g_cost < visited[next_pos]:
                    visited[next_pos] = new_g_cost
                    new_path = path + [direction]
                    
                    # Calculate heuristic to goal
                    new_h_cost = self._heuristic(next_pos, goal)
                    new_f_cost = new_g_cost + new_h_cost
                    
                    heapq.heappush(frontier, (new_f_cost, next_pos, new_path))
                    print(f"  -> {next_pos} via {direction} (cost={move_cost}), g={new_g_cost}, h={new_h_cost}, f={new_f_cost}")
        
        print("No path found to get adjacent to goal!")
        return None

    def _is_adjacent_to_box(self, pos, box_pos):
        """Check if position is adjacent to a box."""
        # Skip if box has been delivered
        if box_pos in self.delivered_boxes:
            return False
            
        for dx, dy in self.directions.values():
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if adj_pos == box_pos:
                return True
        return False

    def _is_adjacent_to_dropzone(self, pos, dropzone_pos):
        """Check if position is adjacent to the dropzone."""
        # Dropzone is always traversable, even after box delivery
        for dx, dy in self.directions.values():
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if adj_pos == dropzone_pos:
                return True
        return False

    def _get_neighbors(self, pos, carrying_box=None, cell_cache=None):
        """Get valid neighboring positions with their costs."""
        neighbors = []
        for direction, (dx, dy) in self.directions.items():
            new_pos = (pos[0] + dx, pos[1] + dy)
            
            # Check if we've already cached this cell
            if cell_cache is not None and new_pos in cell_cache:
                cell = cell_cache[new_pos]
            else:
                try:
                    cell = self.warehouse_viewer[new_pos[0]][new_pos[1]]
                    if cell_cache is not None:
                        cell_cache[new_pos] = cell
                except IndexError:
                    continue  # Skip if position is out of bounds
            
            if self._is_valid_move_cell(cell, carrying_box):
                cost = self.movement_costs[direction]
                neighbors.append((new_pos, direction, cost))
        return neighbors

    def _is_valid_move_cell(self, cell, carrying_box=None):
        """Check if a cell is valid for movement."""
        # If carrying a box, we can move through any non-wall space
        if carrying_box:
            return cell != '#'
        # If not carrying a box, we can only move through empty spaces and dropzone
        return cell in ['.', '@']

    def _get_direction_to_target(self, pos, target):
        """Get the direction from pos to target."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Normalize to -1, 0, or 1
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
            
        # Map to direction string
        if dx == -1 and dy == 0: return 'n'
        if dx == 1 and dy == 0: return 's'
        if dx == 0 and dy == 1: return 'e'
        if dx == 0 and dy == -1: return 'w'
        if dx == -1 and dy == 1: return 'ne'
        if dx == -1 and dy == -1: return 'nw'
        if dx == 1 and dy == 1: return 'se'
        if dx == 1 and dy == -1: return 'sw'
        return None

    def plan_delivery(self, debug=False):
        """
        plan_delivery() is required and will be called by the autograder directly.
        You may not change the method signature for it.
        All print outs must be conditioned on the debug flag.
        """

        # The following is the hard coded solution to test case 1
        # moves = ['move w',
        #          'move nw',
        #          'lift 1',
        #          'move se',
        #          'down e',
        #          'move ne',
        #          'lift 2',
        #          'down s']

        moves = []
        current_pos = self.dropzone_location
        boxes_to_deliver = self.todo.copy()  # Keep original order
        
        if debug:
            print(f"Starting at dropzone: {current_pos}")
            print(f"Boxes to deliver in order: {boxes_to_deliver}")
        
        # Process boxes in the exact order specified by todo
        for next_box in boxes_to_deliver:
            box_pos = self.box_locations[next_box]
            
            if debug:
                print(f"\nPlanning path to box {next_box} at {box_pos}")
            
            # Check if we're already adjacent to box
            if self._is_adjacent_to_box(current_pos, box_pos):
                if debug:
                    print("Already adjacent to box, no movement needed")
                path_to_box = []
            else:
                path_to_box = self._a_star_search(current_pos, box_pos)
                if not path_to_box:
                    if debug:
                        print(f"No path found to box {next_box}")
                    break
            
            # Step 2: Move to box (or adjacent to box)
            for direction in path_to_box:
                moves.append(f'move {direction}')
                # Update position after each move
                dx, dy = self.directions[direction]
                current_pos = (current_pos[0] + dx, current_pos[1] + dy)
                # if debug:
                #     print(f"Move {direction}: Robot at {current_pos}")
            
            # Step 3: Lift box (position doesn't change)
            moves.append(f'lift {next_box}')
            if next_box == '1':
                # pdb.set_trace()
                pass
            box_pickup_location = current_pos  # Store where we picked up the box
            # Update warehouse state to remove the box
            self.warehouse_viewer[box_pos[0]][box_pos[1]] = '.'
            if debug:
                print(f"Lift {next_box}: Robot still at {current_pos}")
                print(f"Updated warehouse state at {box_pos} to '.'")
            
            # Step 4: Find path from current position to dropzone (or adjacent to dropzone)
            if debug:
                print(f"\nPlanning path to dropzone at {self.dropzone_location}")
            
            # If we're on the dropzone, we need to move back to where the box was
            if current_pos == self.dropzone_location:
                if debug:
                    print("Currently on dropzone, need to move back to box location")
                # Calculate direction to move back to box location
                drop_direction = self._get_direction_to_target(current_pos, box_pos)
                if drop_direction:
                    moves.append(f'move {drop_direction}')
                    dx, dy = self.directions[drop_direction]
                    current_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    # if debug:
                    #     print(f"Move {drop_direction}: Robot at {current_pos}")
                path_to_dropzone = []
            # Check if we're already adjacent to dropzone
            elif self._is_adjacent_to_dropzone(current_pos, self.dropzone_location):
                if debug:
                    print("Already adjacent to dropzone, no movement needed")
                path_to_dropzone = []
            else:
                path_to_dropzone = self._a_star_search(current_pos, self.dropzone_location, carrying_box=next_box)
                if not path_to_dropzone:
                    if debug:
                        print(f"No path found to dropzone from box {next_box}")
                    break
            
            # Step 5: Move to dropzone (or adjacent to dropzone)
            for direction in path_to_dropzone:
                moves.append(f'move {direction}')
                # Update position after each move
                dx, dy = self.directions[direction]
                current_pos = (current_pos[0] + dx, current_pos[1] + dy)
                # if debug:
                #     print(f"Move {direction}: Robot at {current_pos}")
            
            # Step 6: Drop box (position doesn't change)
            drop_direction = self._get_direction_to_target(current_pos, self.dropzone_location)
            moves.append(f'down {drop_direction}')
            if debug:
                print(f"Down {drop_direction}: Robot still at {current_pos}")
                print(f"Box {next_box} delivered to dropzone")
            
            # Step 7: Mark box as delivered
            self.delivered_boxes.add(box_pos)
        
        if debug:
            print("\nFinal move list:")
            for move in moves:
                print(move)
            pass
        
        return moves


class DeliveryPlanner_PartB:
    """
    Note: All print outs must be conditioned on the debug parameter.

    Required methods in this class are:

        generate_policies(self, debug = False):
         Stubbed out below. You may not change the method signature
         as it will be called directly by the autograder but you
         may modify the internals as needed.

        __init__:
         Required to initialize the class.  Signature can NOT be changed.
         Basic template starter code is provided.  You may choose to
         use this starter code or modify and replace it based on
         your own solution.

    The following method is starter code you may use.
    However, it is not required and can be replaced with your
    own method(s).

        _set_initial_state_from(self, warehouse):
         creates structures based on the warehouse map

    """

    def __init__(self, warehouse, warehouse_cost, todo):
        self._set_initial_state_from(warehouse)
        self.warehouse_cost = warehouse_cost
        self.todo = todo
        
        # You may use these symbols indicating direction for visual debugging
        # ['^', '<', 'v', '>', '\\', '/', '[', ']']
        # or you may choose to use arrows instead
        # ['🡑', '🡐', '🡓', '🡒',  '🡔', '🡕', '🡖', '🡗']
        
        # Define movement costs and directions
        self.movement_costs = {
            'n': 2, 's': 2, 'e': 2, 'w': 2,  # Cardinal directions
            'ne': 3, 'nw': 3, 'se': 3, 'sw': 3  # Diagonal directions
        }
        
        # Define direction vectors
        self.directions = {
            'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1),
            'ne': (-1, 1), 'nw': (-1, -1), 'se': (1, 1), 'sw': (1, -1)
        }

    def _set_initial_state_from(self, warehouse):
        """Set initial state.

        Args:
            warehouse(list(list)): the warehouse map.
        """
        rows = len(warehouse)
        cols = len(warehouse[0])

        self.warehouse_state = [[None for j in range(cols)] for i in range(rows)]
        self.dropzone = None
        self.boxes = dict()

        for i in range(rows):
            for j in range(cols):
                this_square = warehouse[i][j]

                if this_square == '.':
                    self.warehouse_state[i][j] = '.'

                elif this_square == '#':
                    self.warehouse_state[i][j] = '#'

                elif this_square == '@':
                    self.warehouse_state[i][j] = '@'
                    self.dropzone = (i, j)

                else:  # a box
                    box_id = this_square
                    self.warehouse_state[i][j] = box_id
                    self.boxes[box_id] = (i, j)

    def _is_valid_move(self, pos):
        """Check if a position is valid (within bounds and not a wall)."""
        # Check if position is within grid bounds
        if pos[0] < 0 or pos[0] >= len(self.warehouse_state) or \
           pos[1] < 0 or pos[1] >= len(self.warehouse_state[0]):
            return False
            
        # Check if position is not a wall
        return self.warehouse_state[pos[0]][pos[1]] != '#'

    def _get_neighbors(self, pos):
        """Get valid neighboring positions with their costs."""
        neighbors = []
        for direction, (dx, dy) in self.directions.items():
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(new_pos):
                move_cost = self.movement_costs[direction]
                floor_cost = self.warehouse_cost[new_pos[0]][new_pos[1]]
                total_cost = move_cost + floor_cost
                neighbors.append((new_pos, direction, total_cost))
        return neighbors

    def _heuristic(self, pos, target):
        """Manhattan distance heuristic with diagonal movement consideration."""
        dx = abs(pos[0] - target[0])
        dy = abs(pos[1] - target[1])
        
        # For diagonal movement, we can move both x and y at once
        diagonal = min(dx, dy)
        straight = max(dx, dy) - diagonal
        
        # Use the actual movement costs from our movement_costs dictionary
        heuristic_cost = diagonal * 3 + straight * 2  # 3 for diagonal, 2 for straight
        
        print(f"  Heuristic: pos={pos}, target={target}, dx={dx}, dy={dy}, diagonal={diagonal}, straight={straight}, cost={heuristic_cost}")
        
        return heuristic_cost

    def _a_star_search(self, start, goal, carrying_box=None):
        """A* search algorithm to find optimal path from start to goal."""
        # Initialize frontier as a priority queue
        frontier = []
        heapq.heappush(frontier, (0, start, []))  # (f_cost, position, path)
        
        # Track visited positions and their costs
        visited = {start: 0}  # position -> g_cost
        
        while frontier:
            _, pos, path = heapq.heappop(frontier)
            
            # Check if we've reached the goal or an adjacent position to the goal
            if pos == goal or (not carrying_box and self._is_adjacent_to_box(pos, goal)) or \
               (carrying_box and self._is_adjacent_to_dropzone(pos, goal)):
                return path
                
            for next_pos, direction, move_cost in self._get_neighbors(pos):
                new_cost = visited[pos] + move_cost
                
                # Only explore if we found a better path or haven't seen this position
                if next_pos not in visited or new_cost < visited[next_pos]:
                    visited[next_pos] = new_cost
                    new_path = path + [direction]
                    f_cost = new_cost + self._heuristic(next_pos, goal)
                    heapq.heappush(frontier, (f_cost, next_pos, new_path))
        
        return None

    def _is_adjacent_to_box(self, pos, box_pos):
        """Check if position is adjacent to a box."""
        for dx, dy in self.directions.values():
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if adj_pos == box_pos:
                return True
        return False

    def _is_adjacent_to_dropzone(self, pos, dropzone_pos):
        """Check if position is adjacent to the dropzone."""
        for dx, dy in self.directions.values():
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if adj_pos == dropzone_pos:
                return True
        return False

    def _get_direction_to_target(self, pos, target):
        """Get the direction from pos to target."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Normalize to -1, 0, or 1
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
            
        # Map to direction string
        if dx == -1 and dy == 0: return 'n'
        if dx == 1 and dy == 0: return 's'
        if dx == 0 and dy == 1: return 'e'
        if dx == 0 and dy == -1: return 'w'
        if dx == -1 and dy == 1: return 'ne'
        if dx == -1 and dy == -1: return 'nw'
        if dx == 1 and dy == 1: return 'se'
        if dx == 1 and dy == -1: return 'sw'
        return None

    def generate_policies(self, debug=False):
        """
        generate_policies() is required and will be called by the autograder directly.
        You may not change the function signature for it.
        All print outs must be conditioned on the debug flag.
        """
        rows = len(self.warehouse_state)
        cols = len(self.warehouse_state[0])
        
        # Initialize policies with -1
        to_box_policy = [['-1' for _ in range(cols)] for _ in range(rows)]
        to_zone_policy = [['-1' for _ in range(cols)] for _ in range(rows)]
        
        # Get box position
        box_pos = self.boxes['1']
        
        # Generate to_box_policy
        for i in range(rows):
            for j in range(cols):
                if self.warehouse_state[i][j] == '#':
                    continue
                    
                if (i, j) == box_pos:
                    to_box_policy[i][j] = 'lift 1'
                elif self._is_adjacent_to_box((i, j), box_pos):
                    to_box_policy[i][j] = 'lift 1'
                else:
                    # Find path from this position to box
                    path = self._a_star_search((i, j), box_pos)
                    if path:
                        # Validate the move is valid
                        direction = path[0]
                        dx, dy = self.directions[direction]
                        new_pos = (i + dx, j + dy)
                        if self._is_valid_move(new_pos):
                            to_box_policy[i][j] = f'move {direction}'
        
        # Generate to_zone_policy
        for i in range(rows):
            for j in range(cols):
                if self.warehouse_state[i][j] == '#':
                    continue
                    
                # If we're on the dropzone, we need to move to where the box was
                if (i, j) == self.dropzone:
                    drop_direction = self._get_direction_to_target((i, j), box_pos)
                    if drop_direction:
                        dx, dy = self.directions[drop_direction]
                        new_pos = (i + dx, j + dy)
                        if self._is_valid_move(new_pos):
                            to_zone_policy[i][j] = f'move {drop_direction}'
                    continue
                    
                # Find path from this position to dropzone
                path = self._a_star_search((i, j), self.dropzone, carrying_box='1')
                if path:
                    # Validate the move is valid
                    direction = path[0]
                    dx, dy = self.directions[direction]
                    new_pos = (i + dx, j + dy)
                    if self._is_valid_move(new_pos):
                        to_zone_policy[i][j] = f'move {direction}'
                elif self._is_adjacent_to_dropzone((i, j), self.dropzone):
                    drop_direction = self._get_direction_to_target((i, j), self.dropzone)
                    if drop_direction:
                        dx, dy = self.directions[drop_direction]
                        new_pos = (i + dx, j + dy)
                        if self._is_valid_move(new_pos):
                            to_zone_policy[i][j] = f'down {drop_direction}'
        
        if debug:
            print("\nTo Box Policy:")
            for row in to_box_policy:
                print(row)
            print("\nDeliver Policy:")
            for row in to_zone_policy:
                print(row)
        
        return (to_box_policy, to_zone_policy)


class DeliveryPlanner_PartC:
    """
    [Doc string same as part B]
    Note: All print outs must be conditioned on the debug parameter.

    Required methods in this class are:

        generate_policies(self, debug = False):
         Stubbed out below. You may not change the method signature
         as it will be called directly by the autograder but you
         may modify the internals as needed.

        __init__:
         Required to initialize the class.  Signature can NOT be changed.
         Basic template starter code is provided.  You may choose to
         use this starter code or modify and replace it based on
         your own solution.

    The following method is starter code you may use.
    However, it is not required and can be replaced with your
    own method(s).

        _set_initial_state_from(self, warehouse):
         creates structures based on the warehouse map

    """

    def __init__(self, warehouse, warehouse_cost, todo, stochastic_probabilities):
        self._set_initial_state_from(warehouse)
        self.warehouse_cost = warehouse_cost
        self.todo = todo
        self.stochastic_probabilities = stochastic_probabilities
        
        # Define movement costs and directions
        self.movement_costs = {
            'n': 2, 's': 2, 'e': 2, 'w': 2,  # Cardinal directions
            'ne': 3, 'nw': 3, 'se': 3, 'sw': 3  # Diagonal directions
        }
        
        # Define direction vectors
        self.directions = {
            'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1),
            'ne': (-1, 1), 'nw': (-1, -1), 'se': (1, 1), 'sw': (1, -1)
        }

    def _set_initial_state_from(self, warehouse):
        """Set initial state."""
        rows = len(warehouse)
        cols = len(warehouse[0])

        self.warehouse_state = [[None for j in range(cols)] for i in range(rows)]
        self.dropzone = None
        self.boxes = dict()

        for i in range(rows):
            for j in range(cols):
                this_square = warehouse[i][j]

                if this_square == '.':
                    self.warehouse_state[i][j] = '.'
                elif this_square == '#':
                    self.warehouse_state[i][j] = '#'
                elif this_square == '@':
                    self.warehouse_state[i][j] = '@'
                    self.dropzone = (i, j)
                else:  # a box
                    box_id = this_square
                    self.warehouse_state[i][j] = box_id
                    self.boxes[box_id] = (i, j)


    def _is_valid_move(self, pos):
        """Check if a position is valid (within bounds and not a wall)."""
        # Check if position is within grid bounds
        if pos[0] < 0 or pos[0] >= len(self.warehouse_state) or \
           pos[1] < 0 or pos[1] >= len(self.warehouse_state[0]):
            return False
            
        # Check if position is not a wall
        return self.warehouse_state[pos[0]][pos[1]] != '#'


    def _get_stochastic_outcomes(self, pos, intended_direction):
        """Get all possible outcomes of a stochastic movement."""
        outcomes = []
        dx, dy = self.directions[intended_direction]
        
        # As intended movement
        new_pos = (pos[0] + dx, pos[1] + dy)
        if self._is_valid_move(new_pos):
            outcomes.append((new_pos, self.movement_costs[intended_direction], 
                           self.stochastic_probabilities['as_intended']))
        else:
            # If intended move is invalid, stay in place with intended probability and incur penalty
            outcomes.append((pos, self.movement_costs[intended_direction] + 100, self.stochastic_probabilities['as_intended']))
        
        # Slanted movements
        if intended_direction in ['n', 's']:
            slanted_dirs = ['ne', 'nw'] if intended_direction == 'n' else ['se', 'sw']
        elif intended_direction in ['e', 'w']:
            slanted_dirs = ['ne', 'se'] if intended_direction == 'e' else ['nw', 'sw']
        else:  # Diagonal movements (ne, nw, se, sw)
            if intended_direction == 'ne':
                slanted_dirs = ['n', 'e']
            elif intended_direction == 'nw':
                slanted_dirs = ['n', 'w']
            elif intended_direction == 'se':
                slanted_dirs = ['s', 'e']
            else:  # sw
                slanted_dirs = ['s', 'w']
            
        for dir in slanted_dirs:
            dx, dy = self.directions[dir]
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(new_pos):
                outcomes.append((new_pos, self.movement_costs[dir], 
                               self.stochastic_probabilities['slanted']))
            else:
                # If slanted move is invalid, stay in place with slanted probability and incur penalty
                outcomes.append((pos, self.movement_costs[dir] + 100, self.stochastic_probabilities['slanted']))
        
        # Sideways movements
        if intended_direction in ['n', 's']:
            sideways_dirs = ['e', 'w']
        elif intended_direction in ['e', 'w']:
            sideways_dirs = ['n', 's']
        else:  # Diagonal movements (ne, nw, se, sw)
            if intended_direction == 'ne':
                sideways_dirs = ['nw', 'se']  # Perpendicular diagonals
            elif intended_direction == 'nw':
                sideways_dirs = ['ne', 'sw']  # Perpendicular diagonals  
            elif intended_direction == 'se':
                sideways_dirs = ['ne', 'sw']  # Perpendicular diagonals
            else:  # sw
                sideways_dirs = ['nw', 'se']  # Perpendicular diagonals
            
        for dir in sideways_dirs:
            dx, dy = self.directions[dir]
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(new_pos):
                outcomes.append((new_pos, self.movement_costs[dir], 
                               self.stochastic_probabilities['sideways']))
            else:
                # If sideways move is invalid, stay in place with sideways probability and incur penalty
                outcomes.append((pos, self.movement_costs[dir] + 100, self.stochastic_probabilities['sideways']))
        
        # Stay in place (remaining probability)
        total_prob = sum(p for _, _, p in outcomes)
        if total_prob < 1.0:
            outcomes.append((pos, 0, 1.0 - total_prob))
            
        return outcomes


    def _get_risk_score(self, pos, direction):
        """Calculate risk score for a movement direction based on adjacent squares."""
        dx, dy = self.directions[direction]
        risk_score = 0
        
        # Check all possible outcomes of this movement
        outcomes = self._get_stochastic_outcomes(pos, direction)
        for next_pos, _, prob in outcomes:
            if self._is_valid_move(next_pos):
                # Add risk based on floor cost and probability
                floor_cost = self.warehouse_cost[next_pos[0]][next_pos[1]]
                if floor_cost == float('inf'):
                    risk_score += prob * 1000  # High risk for infinite cost
                else:
                    risk_score += prob * floor_cost
            else:
                risk_score += prob * 100  # Risk for invalid moves
        
        return risk_score


    def _value_iteration(self, goal_pos, is_to_box=True):
        """Value iteration to find optimal policy with stochastic movements."""
        rows = len(self.warehouse_state)
        cols = len(self.warehouse_state[0])
        
        # Initialize values and policy
        values = [[100 for _ in range(cols)] for _ in range(rows)]
        policy = [['-1' for _ in range(cols)] for _ in range(rows)]
        
        # Set goal value to 0
        values[goal_pos[0]][goal_pos[1]] = 0
        
        # Value iteration
        iteration = 0
        max_iterations = 100  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            delta = 0
            
            for i in range(rows):
                for j in range(cols):
                    if i == 0 and j == 1:
                        # pdb.set_trace()
                        pass
                        
                    if self.warehouse_state[i][j] == '#':
                        continue
                    
                    if (i, j) == goal_pos:
                        continue
                    
                    # If adjacent to dropzone and not going to box, set to drop
                    if not is_to_box and self._is_adjacent_to_dropzone((i, j), goal_pos):
                        drop_direction = self._get_direction_to_target((i, j), goal_pos)
                        if drop_direction:
                            policy[i][j] = f'down {drop_direction}'
                            values[i][j] = 0
                            continue
                    
                    old_value = values[i][j]
                    min_value = float('inf')
                    best_action = None
                    
                    # Try all possible actions
                    for direction in self.directions.keys():
                        if direction == 'se' and i == 0 and j == 1:
                            # pdb.set_trace()
                            pass

                        expected_value = 0
                        outcomes = self._get_stochastic_outcomes((i, j), direction)

                        for next_pos, cost, prob in outcomes:
                            # All costs (including penalties) are now handled in _get_stochastic_outcomes
                            floor_cost = self.warehouse_cost[next_pos[0]][next_pos[1]]
                            if floor_cost == float('inf'):
                                expected_value += prob * (cost + values[i][j])  # Use current state value for infinite floor cost
                                continue
                                
                            expected_value += prob * (cost + floor_cost + values[next_pos[0]][next_pos[1]])
                        
                        if expected_value < min_value:
                            min_value = expected_value
                            best_action = f'move {direction}'

                        if i == 0 and j == 1:
                            pass
                            # print(f"expected_value: {expected_value} and direction: {direction}")

                    if min_value < float('inf'):
                        values[i][j] = min_value
                        policy[i][j] = best_action
                    else:
                        # If no valid action found, use a default move action
                        policy[i][j] = 'move n'  # Default action
                    
                    delta = max(delta, abs(old_value - values[i][j]))
            
            if delta < 0.00001:  # Convergence threshold
                break
                
            if iteration % 10 == 0:
                print(f"Value iteration {iteration}, delta: {delta}")
        
        if iteration >= max_iterations:
            print("WARNING: Reached maximum iterations without convergence")
        
        return policy, values


    def _policy_iteration(self, goal_pos, is_to_box=True):
        """Policy iteration to find optimal policy with stochastic movements."""
        rows = len(self.warehouse_state)
        cols = len(self.warehouse_state[0])
        
        # Initialize values and policy
        values = [[100 for _ in range(cols)] for _ in range(rows)]
        policy = [['-1' for _ in range(cols)] for _ in range(rows)]
        
        # Set goal value to 0
        values[goal_pos[0]][goal_pos[1]] = 0
        
        print("\nInitializing policy...")
        # Initialize policy with a default action for each state
        for i in range(rows):
            for j in range(cols):
                if self.warehouse_state[i][j] == '#':
                    continue
                if (i, j) == goal_pos:
                    continue
                    
                # If adjacent to dropzone and not going to box, set to drop
                if not is_to_box and self._is_adjacent_to_dropzone((i, j), goal_pos):
                    drop_direction = self._get_direction_to_target((i, j), goal_pos)
                    if drop_direction:
                        policy[i][j] = f'down {drop_direction}'
                        continue
                
                # Otherwise, set to move in the direction of the goal
                direction = self._get_direction_to_target((i, j), goal_pos)
                if direction:
                    policy[i][j] = f'move {direction}'
                else:
                    policy[i][j] = 'move n'  # Default action
        
        print("Initial policy:")
        for row in policy:
            print(row)
        
        iteration = 0
        max_iterations = 50  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}")
            
            # Policy Evaluation
            eval_iteration = 0
            while eval_iteration < 50:  # Limit evaluation iterations
                eval_iteration += 1
                delta = 0
                for i in range(rows):
                    for j in range(cols):
                        if self.warehouse_state[i][j] == '#':
                            continue
                        if (i, j) == goal_pos:
                            continue
                            
                        old_value = values[i][j]
                        
                        # If policy is 'down', value is 0
                        if policy[i][j].startswith('down'):
                            values[i][j] = 0
                            continue
                            
                        # Get action from policy
                        action = policy[i][j]
                        if action == '-1':
                            continue
                            
                        direction = action.split()[1]
                        expected_value = 0
                        outcomes = self._get_stochastic_outcomes((i, j), direction)
                        
                        for next_pos, cost, prob in outcomes:
                            # All costs (including penalties) are now handled in _get_stochastic_outcomes
                            floor_cost = self.warehouse_cost[next_pos[0]][next_pos[1]]
                            if floor_cost == float('inf'):
                                expected_value += prob * (cost + values[i][j])  # Use current state value for infinite floor cost
                                continue
                                
                            expected_value += prob * (cost + floor_cost + values[next_pos[0]][next_pos[1]])
                        
                        values[i][j] = expected_value
                        delta = max(delta, abs(old_value - values[i][j]))
                
                if delta < 0.00001:  # Convergence threshold
                    break
                    
                if eval_iteration % 100 == 0:
                    print(f"Policy evaluation iteration {eval_iteration}, delta: {delta}")
            
            # Policy Improvement
            policy_stable = True
            changes = 0
            for i in range(rows):
                for j in range(cols):
                    if self.warehouse_state[i][j] == '#':
                        continue
                    if (i, j) == goal_pos:
                        continue
                        
                    # If adjacent to dropzone and not going to box, keep drop action
                    if not is_to_box and self._is_adjacent_to_dropzone((i, j), goal_pos):
                        drop_direction = self._get_direction_to_target((i, j), goal_pos)
                        if drop_direction:
                            old_action = policy[i][j]
                            policy[i][j] = f'down {drop_direction}'
                            if old_action != policy[i][j]:
                                policy_stable = False
                                changes += 1
                            continue
                    
                    # Try all possible actions
                    min_value = float('inf')
                    best_action = None
                    
                    for direction in self.directions.keys():
                        expected_value = 0
                        outcomes = self._get_stochastic_outcomes((i, j), direction)
                        
                        for next_pos, cost, prob in outcomes:
                            # All costs (including penalties) are now handled in _get_stochastic_outcomes
                            floor_cost = self.warehouse_cost[next_pos[0]][next_pos[1]]
                            if floor_cost == float('inf'):
                                expected_value += prob * (cost + values[i][j])  # Use current state value for infinite floor cost
                                continue
                                
                            expected_value += prob * (cost + floor_cost + values[next_pos[0]][next_pos[1]])
                        
                        if expected_value < min_value:
                            min_value = expected_value
                            best_action = f'move {direction}'
                    
                    if best_action and best_action != policy[i][j]:
                        policy[i][j] = best_action
                        policy_stable = False
                        changes += 1
            
            print(f"Policy changes in this iteration: {changes}")
            if policy_stable:
                print("Policy is stable, breaking")
                break
            
            if iteration % 5 == 0:
                print("\nCurrent policy:")
                for row in policy:
                    print(row)
                print("\nCurrent values:")
                for row in values:
                    print(row)
        
        if iteration >= max_iterations:
            print("WARNING: Reached maximum iterations without convergence")
        
        return policy, values

    def _is_adjacent_to_dropzone(self, pos, dropzone_pos):
        """Check if position is adjacent to the dropzone."""
        for dx, dy in self.directions.values():
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if adj_pos == dropzone_pos:
                return True
        return False


    def _get_direction_to_target(self, pos, target):
        """Get the direction from pos to target."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Normalize to -1, 0, or 1
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
            
        # Map to direction string
        if dx == -1 and dy == 0: return 'n'
        if dx == 1 and dy == 0: return 's'
        if dx == 0 and dy == 1: return 'e'
        if dx == 0 and dy == -1: return 'w'
        if dx == -1 and dy == 1: return 'ne'
        if dx == -1 and dy == -1: return 'nw'
        if dx == 1 and dy == 1: return 'se'
        if dx == 1 and dy == -1: return 'sw'
        return None


    def generate_policies(self, debug=False):
        """Generate policies for getting to box and delivering to dropzone."""
        # Get box position
        box_pos = self.boxes['1']
        
        # Generate policy to get to box
        to_box_policy, to_box_values = self._policy_iteration(box_pos, is_to_box=True)
        
        # Mark box position and adjacent cells as 'lift 1'
        to_box_policy[box_pos[0]][box_pos[1]] = 'lift 1'
        for dx, dy in self.directions.values():
            adj_pos = (box_pos[0] + dx, box_pos[1] + dy)
            if self._is_valid_move(adj_pos):
                to_box_policy[adj_pos[0]][adj_pos[1]] = 'lift 1'
        
        # Generate policy to deliver box
        to_zone_policy, to_zone_values = self._policy_iteration(self.dropzone, is_to_box=False)
        
        if debug:
            print("\nTo Box Policy:")
            for row in to_box_policy:
                print(row)
            print("\nTo Zone Policy:")
            for row in to_zone_policy:
                print(row)
            print("\nTo Box Values:")
            for row in to_box_values:
                print(row)
            print("\nTo Zone Values:")
            for row in to_zone_values:
                print(row)
        
        return (to_box_policy, to_zone_policy, to_box_values, to_zone_values)


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami


if __name__ == "__main__":
    """
    You may execute this file to develop and test the search algorithm prior to running
    the delivery planner in the testing suite.  Copy any test cases from the
    testing suite or make up your own.
    Run command:  python warehouse.py
    """

    # Test code in here will NOT be called by the autograder
    # This section is just a provided as a convenience to help in your development/debugging process

    # Testing for Part A
    # print('\n~~~ Testing for part A: ~~~\n')

    # from testing_suite_partA import wrap_warehouse_object, Counter

    # test case data starts here
    # testcase 1
    # warehouse = [
    #     '######',
    #     '#....#',
    #     '#.1#2#',
    #     '#..#.#',
    #     '#...@#',
    #     '######',
    # ]

    # todo = list('12')
    # benchmark_cost = 23
    # viewed_cell_count_threshold = 20
    # dropzone = (4,4)
    # box_locations = {
    #     '1': (2,2),
    #     '2': (2,4),
    # }
    # # test case data ends here

    # The following is the hard coded solution to test case 1
    # moves = ['move w',
    #          'move nw',
    #          'lift 1',
    #          'move se',
    #          'down e',
    #          'move ne',
    #          'lift 2',
    #          'down s']

    # viewed_cells = Counter()
    # warehouse_access = wrap_warehouse_object(warehouse, viewed_cells)
    # partA = DeliveryPlanner_PartA(warehouse_access, dropzone, todo, box_locations)
    # partA.plan_delivery(debug=True)
    # # Note that the viewed cells for the hard coded solution provided
    # # in the initial template code will be 0 because no actual search
    # # process took place that accessed the warehouse
    # print('Viewed Cells:', len(viewed_cells))
    # print('Viewed Cell Count Threshold:', viewed_cell_count_threshold)

    # testcase 4
    # warehouse = [
    #     '########',
    #     '#5######',
    #     '#I#234J#',
    #     '#H#1##6#',
    #     '#G#0@#7#',
    #     '#F####8#',
    #     '#EDCBA9#',
    #     '########'
    #               ]
    
    # todo = list('01234J6789ABCDEFGHI5')
    # benchmark_cost = 636
    # viewed_cell_count_threshold = 59
    # dropzone = (4, 4)
    # box_locations = {
    #     '0': (4, 3),
    #     '1': (3, 3),
    #     '2': (2, 3),
    #     '3': (2, 4),
    #     '4': (2, 5),
    #     '5': (1, 1),
    #     '6': (3, 6),
    #     '7': (4, 6),
    #     '8': (5, 6),
    #     '9': (6, 6),
    #     'A': (6, 5),
    #     'B': (6, 4),
    #     'C': (6, 3),
    #     'D': (6, 2),
    #     'E': (6, 1),
    #     'F': (5, 1),
    #     'G': (4, 1),
    #     'H': (3, 1),
    #     'I': (2, 1),
    #     'J': (2, 6)
    # }

    # viewed_cells = Counter()
    # warehouse_access = wrap_warehouse_object(warehouse, viewed_cells)
    # partA = DeliveryPlanner_PartA(warehouse_access, dropzone, todo, box_locations)
    # partA.plan_delivery(debug=True)
    # # Note that the viewed cells for the hard coded solution provided
    # # in the initial template code will be 0 because no actual search
    # # process took place that accessed the warehouse
    # print('Viewed Cells:', len(viewed_cells))
    # print('Viewed Cell Count Threshold:', viewed_cell_count_threshold)

    # Testing for Part B
    # testcase 1
    # print('\n~~~ Testing for part B: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[3, 5, 2],
    #                   [10, math.inf, 2],
    #                   [2, 10, 2]]

    # todo = ['1']

    # partB = DeliveryPlanner_PartB(warehouse, warehouse_cost, todo)
    # partB.generate_policies(debug=True)

    # Testing for Part C
    # print('\n~~~ Testing for part C: ~~~')

    # testcase 1
    # print('\n~~~ Testing for part C test case 1: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .70,
    #     'slanted': .1,
    #     'sideways': .05,
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # testcase 2
    # print('\n~~~ Testing for part C test case 2: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .20,  # Note: lower probability of moving as intended
    #     'slanted': ((1 - .20) / 3),  # prob_not_as_intended / 3 
    #     'sideways': ((1 - .20) / 6),  # prob_not_as_intended / 6
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # testcase 3
    # print('\n~~~ Testing for part C test case 3: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .60,
    #     'slanted': ((1 - .60) / 3),
    #     'sideways': ((1 - .60) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # # testcase 4
    # print('\n~~~ Testing for part C test case 4: ~~~')
    # warehouse = ['.........#..........',
    #              '...#.....#..........',
    #              '1..#................',
    #              '...#................',
    #              '....#....#####....##',
    #              '......#..#..........',
    #              '......#..#...@......']

    # warehouse_cost = [[94, 56, 14, 0, 11, 74, 4, 85, 88, math.inf, 10, 12, 93, 45, 30, 2, 3, 95, 2, 44],
    #                   [82, 79, 61, math.inf, 78, 59, 19, 11, 23, math.inf, 91, 14, 1, 64, 62, 31, 8, 85, 69, 59],
    #                   [0, 8, 76, math.inf, 86, 11, 65, 74, 5, 34, 71, 8, 82, 38, 61, 45, 34, 31, 83, 25],
    #                   [58, 67, 85, math.inf, 2, 65, 9, 0, 42, 18, 90, 60, 84, 48, 21, 6, 9, 75, 63, 20],
    #                   [9, 71, 27, 18, math.inf, 3, 44, 93, 14, math.inf, math.inf, math.inf, math.inf, math.inf, 67, 18, 85, 39, math.inf, math.inf],
    #                   [58, 5, 53, 35, 84, 5, math.inf, 22, 34, math.inf, 19, 38, 19, 94, 59, 5, 72, 49, 92, 44],
    #                   [63, 43, 74, 59, 60, 5, math.inf, 95, 60, math.inf, 76, 21, 56, 0, 93, 94, 66, 56, 37, 35]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .80,
    #     'slanted': ((1 - .80) / 3),
    #     'sideways': ((1 - .80) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # # testcase 5
    # print('\n~~~ Testing for part C test case 5: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .50,
    #     'slanted': ((1 - .50) / 3),
    #     'sideways': ((1 - .50) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # # testcase 6
    # print('\n~~~ Testing for part C test case 6: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .90,
    #     'slanted': ((1 - .90) / 3),
    #     'sideways': ((1 - .90) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # # testcase 7
    # print('\n~~~ Testing for part C test case 7: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .30,
    #     'slanted': ((1 - .30) / 3),
    #     'sideways': ((1 - .30) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # # testcase 8
    # print('\n~~~ Testing for part C test case 8: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .10,
    #     'slanted': ((1 - .10) / 3),
    #     'sideways': ((1 - .10) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # # testcase 9
    # print('\n~~~ Testing for part C test case 9: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .05,
    #     'slanted': ((1 - .05) / 3),
    #     'sideways': ((1 - .05) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # # testcase 10
    # print('\n~~~ Testing for part C test case 10: ~~~')
    # warehouse = ['1..',
    #              '.#.',
    #              '..@']

    # warehouse_cost = [[13, 5, 6],
    #                   [10, math.inf, 2],
    #                   [2, 11, 2]]

    # todo = ['1']

    # stochastic_probabilities = {
    #     'as_intended': .01,
    #     'slanted': ((1 - .01) / 3),
    #     'sideways': ((1 - .01) / 6),
    # }

    # partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    # partC.generate_policies(debug=True)

    # test case data starts here
    # testcase 10
    # warehouse = [
    #     '#######################',
    #     '#........#####.......@#',
    #     '#.......##...##.......#',
    #     '#.....###.....###.....#',
    #     '#....##..#...#..##....#',
    #     '#..##.............##..#',
    #     '#...##..#.....#..##...#',
    #     '#...##...#...#...##...#',
    #     '#....#....###....#....#',
    #     '#....#..........##....#',
    #     '#.....###########.....#',
    #     '#12...................#',
    #     '#######################'
    # ]

    # todo = list('21')
    # benchmark_cost = 1097
    # viewed_cell_count_threshold = 10211
    # dropzone = (1, 21)
    # box_locations = {
    #     '1': (11, 1),
    #     '2': (11, 2),
    #     '3': (10, 1),
    #     '4': (10, 2)
    # }

    # viewed_cells = Counter()
    # warehouse_access = wrap_warehouse_object(warehouse, viewed_cells)
    # partA = DeliveryPlanner_PartA(warehouse_access, dropzone, todo, box_locations)
    # partA.plan_delivery(debug=True)
    # print('Viewed Cells:', len(viewed_cells))
    # print('Viewed Cell Count Threshold:', viewed_cell_count_threshold)

    # --- Only run Part A test case 7 ---
    print('\n~~~ Testing for part A: Test Case 7 Only ~~~\n')
    from testing_suite_partA import wrap_warehouse_object, Counter
    warehouse = [
        '########',
        '#......#',
        '#....1.#',
        '#......#',
        '#......#',
        '#.....@#',
        '########'
    ]
    todo = list('1')
    benchmark_cost = 12
    viewed_cell_count_threshold = 23
    dropzone = (5, 6)
    box_locations = {
        '1': (2, 5)
    }
    viewed_cells = Counter()
    warehouse_access = wrap_warehouse_object(warehouse, viewed_cells)
    partA = DeliveryPlanner_PartA(warehouse_access, dropzone, todo, box_locations)
    partA.plan_delivery(debug=True)
    print('Viewed Cells:', len(viewed_cells))
    print('Viewed Cell Count Threshold:', viewed_cell_count_threshold)
