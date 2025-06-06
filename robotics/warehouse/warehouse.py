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
        
        self.directions = {
            'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1),
            'ne': (-1, 1), 'nw': (-1, -1), 'se': (1, 1), 'sw': (1, -1)
        }


    def _is_valid_move(self, pos):
        """Check if a position is valid (not a wall)."""
        try:
            return self.warehouse_viewer[pos[0]][pos[1]] != '#'
        except:
            return False


    def _get_neighbors(self, pos, carrying_box=None):
        """Get valid neighboring positions."""
        neighbors = []
        for direction, (dx, dy) in self.directions.items():
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(new_pos):
                neighbors.append((new_pos, direction))
        return neighbors


    def _heuristic(self, pos, target):
        """Manhattan distance heuristic."""
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])


    def _a_star_search(self, start, goal, carrying_box=None):
        """A* search algorithm to find path from start to goal."""
        frontier = [(0, start, [])]  # (cost, position, path)
        visited = {start}
        
        while frontier:
            cost, pos, path = frontier.pop(0)
            
            if pos == goal:
                return path
                
            for next_pos, direction in self._get_neighbors(pos, carrying_box):
                if next_pos not in visited:
                    visited.add(next_pos)
                    new_path = path + [direction]
                    new_cost = len(new_path) + self._heuristic(next_pos, goal)
                    frontier.append((new_cost, next_pos, new_path))
                    frontier.sort()  # Sort by cost
        
        return None


    def plan_delivery(self, debug=False):
        """
        plan_delivery() is required and will be called by the autograder directly.
        You may not change the function signature for it.
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
        
        while boxes_to_deliver:
            # Find closest box
            closest_box = None
            min_dist = float('inf')
            closest_path = None
            
            for box in boxes_to_deliver:
                box_pos = self.box_locations[box]
                path = self._a_star_search(current_pos, box_pos)
                if path and len(path) < min_dist:
                    min_dist = len(path)
                    closest_box = box
                    closest_path = path
            
            if not closest_box:
                break
                
            # Move to box
            for direction in closest_path:
                moves.append(f'move {direction}')
            
            # Lift box
            moves.append(f'lift {closest_box}')
            current_pos = self.box_locations[closest_box]
            
            # Move to dropzone
            path_to_dropzone = self._a_star_search(current_pos, self.dropzone_location)
            if not path_to_dropzone:
                break
                
            for direction in path_to_dropzone:
                moves.append(f'move {direction}')
            
            # Drop box
            moves.append(f'down {direction}')
            current_pos = self.dropzone_location
            boxes_to_deliver.remove(closest_box)
        
        if debug:
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


    def generate_policies(self, debug=False):
        """
        generate_policies() is required and will be called by the autograder directly.
        You may not change the function signature for it.
        All print outs must be conditioned on the debug flag.
        """

        # The following is the hard coded solution to test case 1
        to_box_policy = [['B', 'lift 1', 'move w'],
                  ['lift 1', '-1', 'move nw'],
                  ['move n', 'move nw', 'move n']]

        deliver_policy = [['move e', 'move se', 'move s'],
                  ['move ne', '-1', 'down s'],
                  ['move e', 'down e', 'move n']]

        if debug:
            print("\nTo Box Policy:")
            for i in range(len(to_box_policy)):
                print(to_box_policy[i])

            print("\nDeliver Policy:")
            for i in range(len(deliver_policy)):
                print(deliver_policy[i])

        return (to_box_policy, deliver_policy)


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


    def generate_policies(self, debug=False):
        """
        generate_policies() is required and will be called by the autograder directly.
        You may not change the function signature for it.
        All print outs must be conditioned on the debug flag.
        """

        # The following is the hard coded solution to test case 1
        to_box_policy = [
            ['B', 'lift 1', 'move w'],
            ['lift 1', -1, 'move nw'],
            ['move n', 'move nw', 'move n'],
        ]

        to_zone_policy = [
            ['move e', 'move se', 'move s'],
            ['move se', -1, 'down s'],
            ['move e', 'down e', 'move n'],
        ]

        if debug:
            print("\nTo Box Policy:")
            for i in range(len(to_box_policy)):
                print(to_box_policy[i])

            print("\nTo Zone Policy:")
            for i in range(len(to_zone_policy)):
                print(to_zone_policy[i])

        # For debugging purposes you may wish to return values associated with each policy.
        # Replace the default values of None with your grid of values below and turn on the
        # VERBOSE_FLAG in the testing suite.
        to_box_values = None
        to_zone_values = None
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

    # Testing for Part B
    # testcase 1
    print('\n~~~ Testing for part B: ~~~')
    warehouse = ['1..',
                 '.#.',
                 '..@']

    warehouse_cost = [[3, 5, 2],
                      [10, math.inf, 2],
                      [2, 10, 2]]

    todo = ['1']

    partB = DeliveryPlanner_PartB(warehouse, warehouse_cost, todo)
    partB.generate_policies(debug=True)

    # Testing for Part C
    # testcase 1
    print('\n~~~ Testing for part C: ~~~')
    warehouse = ['1..',
                 '.#.',
                 '..@']

    warehouse_cost = [[13, 5, 6],
                      [10, math.inf, 2],
                      [2, 11, 2]]

    todo = ['1']

    stochastic_probabilities = {
        'as_intended': .70,
        'slanted': .1,
        'sideways': .05,
    }

    partC = DeliveryPlanner_PartC(warehouse, warehouse_cost, todo, stochastic_probabilities)
    partC.generate_policies(debug=True)
