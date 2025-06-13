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

    def _is_valid_move(self, pos, carrying_box=None):
        """Check if a position is valid (not a wall)."""
        try:
            cell = self.warehouse_viewer[pos[0]][pos[1]]
            # If carrying a box, we can move through any non-wall space
            if carrying_box:
                return cell != '#'
            # If not carrying a box, we can move through empty spaces and the dropzone
            return cell in ['.', '@']
        except IndexError:
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


    def _heuristic(self, pos, target):
        """Manhattan distance heuristic with diagonal movement consideration."""
        dx = abs(pos[0] - target[0])
        dy = abs(pos[1] - target[1])
        
        # For diagonal movement, we can move both x and y at once
        diagonal = min(dx, dy)
        straight = max(dx, dy) - diagonal
        
        # Use the actual movement costs from our movement_costs dictionary
        return diagonal * 3 + straight * 2  # 3 for diagonal, 2 for straight


    def _a_star_search(self, start, goal, carrying_box=None):
        """A* search algorithm to find optimal path from start to goal."""
        import heapq
        
        # Initialize frontier as a priority queue
        frontier = []
        heapq.heappush(frontier, (0, start, []))  # (f_cost, position, path)
        
        # Track visited positions and their costs
        visited = {start: 0}  # position -> g_cost
        
        # Cache for cell contents to avoid repeated viewing
        cell_cache = {start: self.warehouse_viewer[start[0]][start[1]]}
        
        while frontier:
            _, pos, path = heapq.heappop(frontier)
            
            # Check if we've reached the goal or an adjacent position to the goal
            if pos == goal or (not carrying_box and self._is_adjacent_to_box(pos, goal, cell_cache)) or \
               (carrying_box and self._is_adjacent_to_dropzone(pos, goal, cell_cache)):
                return path
                
            for next_pos, direction, move_cost in self._get_neighbors(pos, carrying_box, cell_cache):
                new_cost = visited[pos] + move_cost
                
                # Only explore if we found a better path or haven't seen this position
                if next_pos not in visited or new_cost < visited[next_pos]:
                    visited[next_pos] = new_cost
                    new_path = path + [direction]
                    f_cost = new_cost + self._heuristic(next_pos, goal)
                    heapq.heappush(frontier, (f_cost, next_pos, new_path))
        
        return None

    def _is_adjacent_to_box(self, pos, box_pos, cell_cache):
        """Check if position is adjacent to a box."""
        # Skip if box has been delivered
        if box_pos in self.delivered_boxes:
            return False
            
        for dx, dy in self.directions.values():
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if adj_pos == box_pos:
                return True
        return False

    def _is_adjacent_to_dropzone(self, pos, dropzone_pos, cell_cache):
        """Check if position is adjacent to the dropzone."""
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
        # If not carrying a box, we can move through empty spaces, dropzone, and boxes
        return cell in ['.', '@'] or cell.isalnum()

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
        boxes_to_deliver = self.todo.copy()
        
        if debug:
            print(f"Starting at dropzone: {current_pos}")
        
        while boxes_to_deliver:
            # Get the next box to deliver
            next_box = boxes_to_deliver[0]
            box_pos = self.box_locations[next_box]
            
            if debug:
                print(f"\nPlanning path to box {next_box} at {box_pos}")
            
            # Step 1: Find path from current position to box (or adjacent to box)
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
                if debug:
                    print(f"Move {direction}: Robot at {current_pos}")
            
            # Step 3: Lift box (position doesn't change)
            moves.append(f'lift {next_box}')
            if debug:
                print(f"Lift {next_box}: Robot still at {current_pos}")
            
            # Step 4: Find path from current position to dropzone (or adjacent to dropzone)
            if debug:
                print(f"\nPlanning path to dropzone at {self.dropzone_location}")
            
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
                if debug:
                    print(f"Move {direction}: Robot at {current_pos}")
            
            # Step 6: Drop box (position doesn't change)
            moves.append(f'down {path_to_dropzone[-1]}')  # Use last direction for dropping
            if debug:
                print(f"Down {path_to_dropzone[-1]}: Robot still at {current_pos}")
            
            # Step 7: Mark box as delivered and remove from todo list
            self.delivered_boxes.add(box_pos)
            boxes_to_deliver.pop(0)
        
        if debug:
            print("\nFinal move list:")
            for move in moves:
                print(move)
        
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
        """Check if a position is valid (not a wall)."""
        try:
            return self.warehouse_state[pos[0]][pos[1]] != '#'
        except:
            return False

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

    def _value_iteration(self, goal_pos, is_to_box=True):
        """Value iteration to find optimal policy."""
        rows = len(self.warehouse_state)
        cols = len(self.warehouse_state[0])
        
        # Initialize values and policy
        values = [[float('inf') for _ in range(cols)] for _ in range(rows)]
        policy = [['-1' for _ in range(cols)] for _ in range(rows)]
        
        # Set goal value to 0
        values[goal_pos[0]][goal_pos[1]] = 0
        
        # Value iteration
        while True:
            delta = 0
            for i in range(rows):
                for j in range(cols):
                    if self.warehouse_state[i][j] == '#':
                        continue
                        
                    if (i, j) == goal_pos:
                        continue
                        
                    old_value = values[i][j]
                    min_value = float('inf')
                    best_action = '-1'
                    
                    # Try all possible actions
                    for next_pos, direction, cost in self._get_neighbors((i, j)):
                        new_value = cost + values[next_pos[0]][next_pos[1]]
                        if new_value < min_value:
                            min_value = new_value
                            best_action = direction
                    
                    if min_value < float('inf'):
                        values[i][j] = min_value
                        policy[i][j] = best_action
                    
                    delta = max(delta, abs(old_value - values[i][j]))
            
            if delta < 0.001:  # Convergence threshold
                break
        
        return policy

    def generate_policies(self, debug=False):
        """
        generate_policies() is required and will be called by the autograder directly.
        You may not change the function signature for it.
        All print outs must be conditioned on the debug flag.
        """

        # The following is the hard coded solution to test case 1
        # to_box_policy = [['B', 'lift 1', 'move w'],
        #           ['lift 1', '-1', 'move nw'],
        #           ['move n', 'move nw', 'move n']]
        # 
        # deliver_policy = [['move e', 'move se', 'move s'],
        #           ['move ne', '-1', 'down s'],
        #           ['move e', 'down e', 'move n']]

        # Get box position
        box_pos = self.boxes['1']
        
        # Generate policy to get to box
        to_box_policy = self._value_iteration(box_pos, is_to_box=True)
        to_box_policy[box_pos[0]][box_pos[1]] = 'lift 1'
        
        # Generate policy to deliver box
        to_zone_policy = self._value_iteration(self.dropzone, is_to_box=False)
        
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
        """Check if a position is valid (not a wall)."""
        try:
            return self.warehouse_state[pos[0]][pos[1]] != '#'
        except:
            return False

    def _get_stochastic_outcomes(self, pos, intended_direction):
        """Get all possible outcomes of a stochastic movement."""
        outcomes = []
        dx, dy = self.directions[intended_direction]
        
        # As intended movement
        new_pos = (pos[0] + dx, pos[1] + dy)
        if self._is_valid_move(new_pos):
            outcomes.append((new_pos, self.movement_costs[intended_direction], 
                           self.stochastic_probabilities['as_intended']))
        
        # Slanted movements
        if intended_direction in ['n', 's']:
            slanted_dirs = ['ne', 'nw'] if intended_direction == 'n' else ['se', 'sw']
        elif intended_direction in ['e', 'w']:
            slanted_dirs = ['ne', 'se'] if intended_direction == 'e' else ['nw', 'sw']
        else:  # Diagonal movements
            slanted_dirs = [intended_direction]
            
        for dir in slanted_dirs:
            dx, dy = self.directions[dir]
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(new_pos):
                outcomes.append((new_pos, self.movement_costs[dir], 
                               self.stochastic_probabilities['slanted']))
        
        # Sideways movements
        if intended_direction in ['n', 's']:
            sideways_dirs = ['e', 'w']
        elif intended_direction in ['e', 'w']:
            sideways_dirs = ['n', 's']
        else:  # Diagonal movements
            sideways_dirs = []
            
        for dir in sideways_dirs:
            dx, dy = self.directions[dir]
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(new_pos):
                outcomes.append((new_pos, self.movement_costs[dir], 
                               self.stochastic_probabilities['sideways']))
        
        # Stay in place (if any movement fails)
        total_prob = sum(p for _, _, p in outcomes)
        if total_prob < 1.0:
            outcomes.append((pos, 0, 1.0 - total_prob))
            
        return outcomes

    def _value_iteration(self, goal_pos, is_to_box=True):
        """Value iteration to find optimal policy with stochastic movements."""
        rows = len(self.warehouse_state)
        cols = len(self.warehouse_state[0])
        
        # Initialize values and policy
        values = [[float('inf') for _ in range(cols)] for _ in range(rows)]
        policy = [['-1' for _ in range(cols)] for _ in range(rows)]
        
        # Set goal value to 0
        values[goal_pos[0]][goal_pos[1]] = 0
        
        # Value iteration
        while True:
            delta = 0
            for i in range(rows):
                for j in range(cols):
                    if self.warehouse_state[i][j] == '#':
                        continue
                        
                    if (i, j) == goal_pos:
                        continue
                        
                    old_value = values[i][j]
                    min_value = float('inf')
                    best_action = '-1'
                    
                    # Try all possible actions
                    for direction in self.directions.keys():
                        expected_value = 0
                        outcomes = self._get_stochastic_outcomes((i, j), direction)
                        
                        for next_pos, cost, prob in outcomes:
                            floor_cost = self.warehouse_cost[next_pos[0]][next_pos[1]]
                            expected_value += prob * (cost + floor_cost + values[next_pos[0]][next_pos[1]])
                        
                        if expected_value < min_value:
                            min_value = expected_value
                            best_action = direction
                    
                    if min_value < float('inf'):
                        values[i][j] = min_value
                        policy[i][j] = best_action
                    
                    delta = max(delta, abs(old_value - values[i][j]))
            
            if delta < 0.001:  # Convergence threshold
                break
        
        return policy, values

    def generate_policies(self, debug=False):
        """
        generate_policies() is required and will be called by the autograder directly.
        You may not change the function signature for it.
        All print outs must be conditioned on the debug flag.
        """

        # The following is the hard coded solution to test case 1
        # to_box_policy = [
        #     ['B', 'lift 1', 'move w'],
        #     ['lift 1', -1, 'move nw'],
        #     ['move n', 'move nw', 'move n'],
        # ]
        # 
        # to_zone_policy = [
        #     ['move e', 'move se', 'move s'],
        #     ['move se', -1, 'down s'],
        #     ['move e', 'down e', 'move n'],
        # ]

        # Get box position
        box_pos = self.boxes['1']
        
        # Generate policy to get to box
        to_box_policy, to_box_values = self._value_iteration(box_pos, is_to_box=True)
        to_box_policy[box_pos[0]][box_pos[1]] = 'lift 1'
        
        # Generate policy to deliver box
        to_zone_policy, to_zone_values = self._value_iteration(self.dropzone, is_to_box=False)
        
        if debug:
            print("\nTo Box Policy:")
            for row in to_box_policy:
                print(row)
            print("\nTo Zone Policy:")
            for row in to_zone_policy:
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
    print('\n~~~ Testing for part A: ~~~\n')

    from testing_suite_partA import wrap_warehouse_object, Counter

    # test case data starts here
    # testcase 1
    warehouse = [
        '######',
        '#....#',
        '#.1#2#',
        '#..#.#',
        '#...@#',
        '######',
    ]
    todo = list('12')
    benchmark_cost = 23
    viewed_cell_count_threshold = 20
    dropzone = (4,4)
    box_locations = {
        '1': (2,2),
        '2': (2,4),
    }
    # test case data ends here

    viewed_cells = Counter()
    warehouse_access = wrap_warehouse_object(warehouse, viewed_cells)
    partA = DeliveryPlanner_PartA(warehouse_access, dropzone, todo, box_locations)
    partA.plan_delivery(debug=True)
    # Note that the viewed cells for the hard coded solution provided
    # in the initial template code will be 0 because no actual search
    # process took place that accessed the warehouse
    print('Viewed Cells:', len(viewed_cells))
    print('Viewed Cell Count Threshold:', viewed_cell_count_threshold)

    # # Testing for Part B
    # # testcase 1
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

    # # Testing for Part C
    # # testcase 1
    # print('\n~~~ Testing for part C: ~~~')
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
