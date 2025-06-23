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

import argparse
import copy
import math
import time
import turtle
from typing import Tuple, Dict
from drone import truncate_angle
import drone

import test_cases
import testing_suite_indiana_drones as testing_suite
import indiana_drones

PI = math.pi

########################################################################
# set to True for lots-o-output, also passed into drone under test
########################################################################
VERBOSE_FLAG = False

class TurtleDisplay(object):
    WIDTH = 800
    HEIGHT = 800

    def __init__(self, xbounds: Tuple[float, float], ybounds: Tuple[float, float]):
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.tree_turtles = {}
        self.tree_estimate_turtles = {}
        self.drone_turtle = None
        self.drone_heading = None
        self.drone_estimate_turtle = None
        self.treasure_turtle = None

    def setup(self):
        xmin, xmax = self.xbounds
        ymin, ymax = self.ybounds
        dx = xmax - xmin
        dy = ymax - ymin
        margin = 0.0
        turtle.setup(width=self.WIDTH,
                     height=self.HEIGHT)
        turtle.setworldcoordinates(xmin - (dx * margin),
                                   ymin - (dy * margin),
                                   xmax + (dx * margin),
                                   ymax + (dy * margin))
        turtle.tracer(0, 1)
        turtle.hideturtle()
        turtle.penup()
        

    def start_time_step(self):

        for tree_id, trtl in self.tree_turtles.items():
            trtl.clear()

    def _new_turtle(self, shape: str = 'circle', color: str = 'gray', shapesize: Tuple[float, float] = (0.1, 0.1)):

        trtl = turtle.Turtle()
        trtl.shape(shape)
        trtl.color(color)
        trtl.shapesize(*shapesize)
        trtl.penup()
        return trtl

    def treasure_at_location(self, treasure_id: str, x: float, y: float):

        key = (treasure_id, x, y)

        if self.treasure_turtle is None:
            self.treasure_turtle = self._new_turtle(shape='triangle',
                                                     color='red',
                                                     shapesize=(2*0.1*4, 2*0.1*4)) # 20 pixels = 1 

        self.treasure_turtle.setposition(x+0.1, y)
        self.treasure_turtle._write(str(treasure_id)[:2], 'left', 'arial')
        self.treasure_turtle.setposition(x, y)

    
    def tree_at_location(self, tree_id: str, x: float, y: float, r: float, col: str):

        key = (tree_id, x, y, r)

        if key not in self.tree_turtles:
            self.tree_turtles[key] = self._new_turtle(shape='circle',
                                                     color='green',
                                                     shapesize=(2*r*4, 2*r*4)) # 20 pixels = 1 
        self.tree_turtles[key].color(col)
        self.tree_turtles[key].setposition(x+r, y)
        self.tree_turtles[key]._write(str(tree_id)[:2], 'left', 'arial')
        self.tree_turtles[key].setposition(x, y)


    def tree_estimate_at_location(self, tree_id: str, x: float, y: float):

        if tree_id not in self.tree_estimate_turtles:
            trtl = turtle.Turtle()
            trtl.shape("circle")
            trtl.color("black" if tree_id != 'self' else 'red')
            trtl.shapesize(0.2, 0.2)
            trtl.penup()
            self.tree_estimate_turtles[tree_id] = trtl

        self.tree_estimate_turtles[tree_id].setposition(x, y)

    def drone_at_location(self, x: float, y: float, bearing: float):
        
        r = 0.15
        
        if self.drone_heading is None:
            self.drone_turtle = self._new_turtle(color='red',
                                                 shapesize=(0.7*r*4, 0.7*r*4))
            self.drone1_turtle = self._new_turtle(color='black',
                                                 shapesize=(1*r*4, 1*r*4))
            self.drone2_turtle = self._new_turtle(color='black',
                                                 shapesize=(1*r*4, 1*r*4))
            self.drone3_turtle = self._new_turtle(color='black',
                                                 shapesize=(1*r*4, 1*r*4))
            self.drone4_turtle = self._new_turtle(color='black',
                                                 shapesize=(1*r*4, 1*r*4))

            self.drone_heading = self._new_turtle(color='red', shape='arrow', shapesize=(0.7*r*4, 1.4*r*4))

        self.drone_turtle.setposition(x, y)
        self.drone1_turtle.setposition(x+r*math.cos(bearing+(math.pi/4)), y+r*math.sin(bearing+(math.pi/4)))
        self.drone2_turtle.setposition(x+r*math.cos(bearing+(3*math.pi/4)), y+r*math.sin(bearing+(3*math.pi/4)))
        self.drone3_turtle.setposition(x+r*math.cos(bearing+(5*math.pi/4)), y+r*math.sin(bearing+(5*math.pi/4)))
        self.drone4_turtle.setposition(x+r*math.cos(bearing+(7*math.pi/4)), y+r*math.sin(bearing+(7*math.pi/4)))
       
        
        self.drone_heading.setposition(x, y)
        self.drone_heading.settiltangle(math.degrees(bearing))
        self.drone_heading.pendown()
        self.drone_heading.stamp()

    def drone_estimate_at_location(self, x: float, y: float):

        if self.drone_estimate_turtle is None:
            self.drone_estimate_turtle = self._new_turtle(color='blue',
                                                          shapesize=(0.3, 0.3))

        self.drone_estimate_turtle.setposition(x, y)
        self.drone_estimate_turtle.pendown()
        self.drone_estimate_turtle.stamp()

    def end_time_step(self):
        turtle.update()
        time.sleep(1.0)

    def done(self):
        turtle.done()

def part_A(params: Dict):
    area_map = params['area_map']
    tree_radius = params['tree_radius']
    horizon = params['horizon']
    state = testing_suite.State(area_map=area_map, tree_radius=tree_radius,
                                measure_distance_noise=params['drone_distance_noise'],
                                measure_bearing_noise=params['drone_bearing_noise'],
                                horizon=horizon)

    if VERBOSE_FLAG:
        print('Initial State:')
        print(state)
    
    drone_init_x = state.drone.x
    drone_init_y = state.drone.y
    drone_init_b = state.drone.bearing

    xbounds, ybounds = (0,10), (-10,0)

    display = TurtleDisplay(xbounds=xbounds,
                            ybounds=ybounds)
    display.setup()

    display.start_time_step()
    display.drone_at_location(drone_init_x, drone_init_y, drone_init_b)
    display.end_time_step()
    display.treasure_at_location(state._treasure_location['type'], state._treasure_location['x'], state._treasure_location['y'])

    try:

        rover_slam = indiana_drones.SLAM()

        for move in params['move']:

            display.start_time_step()

            meas = state.generate_measurements()
            rover_slam.process_measurements(meas)

            action = move.split()
            state.update_according_to(move)

            if VERBOSE_FLAG:
                print(state)

            rover_slam.process_movement(float(action[1]), float(action[2]))
            coordinates = rover_slam.get_coordinates()
            belief_x, belief_y = coordinates['self']

            belief = (belief_x + drone_init_x, belief_y + drone_init_y)
            truth = (state.drone.x - state._start_position['x'] + drone_init_x,
                     state.drone.y - state._start_position['y'] + drone_init_y)
            drone_dist_error = drone.compute_distance(belief, truth)


            if VERBOSE_FLAG:
                    print("Current Belief:", belief)
                    print("True Position:", truth)
                    print("Error:", drone_dist_error, "\n")

            for tree in state.tree_locs_on_map:
                if tree['id'] in meas:
                    col = 'green'
                else:
                    col = 'gray'
                display.tree_at_location(tree['type'], tree['x'], tree['y'], tree['radius'], col)

            display.drone_at_location(*truth, state.drone.bearing)
            display.drone_estimate_at_location(*belief)

            for location in state.tree_locs_on_map:
                if len(coordinates) > 0 and location['id'] in coordinates:
                    x, y = coordinates[location['id']]
                    display.tree_estimate_at_location(location['id'],
                                                    x + drone_init_x,
                                                    y + drone_init_y)

            display.end_time_step()
        
        if VERBOSE_FLAG:
                print('Final State:')
                print(state)

    except Exception as e:
        print(e)

    turtle.bye()


def part_B(params: Dict):
    area_map = params['area_map']
    tree_radius = params['tree_radius']
    max_distance = params['max_distance']
    max_steering = params['max_steering']
    horizon = params['horizon']
    state = testing_suite.State(area_map=area_map,
                                tree_radius=tree_radius,
                                max_distance=max_distance,
                                max_steering=max_steering,
                                measure_distance_noise=params['drone_distance_noise'],
                                measure_bearing_noise=params['drone_bearing_noise'],
                                horizon=horizon)
    
    if VERBOSE_FLAG:
        print('Initial State:')
        print(state)

    drone_init_x = state.drone.x
    drone_init_y = state.drone.y

    xbounds, ybounds = (0,10), (-10,0)

    display = TurtleDisplay(xbounds=xbounds,
                            ybounds=ybounds)
    display.setup()

    # display initial state
    display.start_time_step()

    for tree in state.tree_locs_on_map:
        display.tree_at_location(tree['type'], tree['x'], tree['y'], tree['radius'],'green')

    display.drone_at_location(state.drone.x, state.drone.y, state.drone.bearing)
    display.treasure_at_location(state._treasure_location['type'], state._treasure_location['x'], state._treasure_location['y'])


    display.end_time_step()

    try:
        student_planner = indiana_drones.IndianaDronesPlanner(max_distance, max_steering)

        while len(state.collected_treasure) < 1:

            display.start_time_step()

            meas = state.generate_measurements()
            ret = student_planner.next_move(meas, state._treasure_loc_fromstart)
            
            try:
                action, locs = ret

            except IndexError:
                action = ret
                locs = None

            state.update_according_to(action)
            if VERBOSE_FLAG:
                print(state)

            for tree in state.tree_locs_on_map:
                if tree['id'] in meas:
                    col = 'green'
                else:
                    col = 'gray'
                display.tree_at_location(tree['type'], tree['x'], tree['y'], tree['radius'], col)

            display.drone_at_location(state.drone.x, state.drone.y, state.drone.bearing)

            if locs:
                for locid, xy in locs.items():
                    x, y = xy
                    if locid == 'self':
                        display.drone_estimate_at_location(x + drone_init_x,
                                                           y + drone_init_y)
                        
                        belief = (x + drone_init_x, y + drone_init_y)
                        truth = (state.drone.x - state._start_position['x'] + drone_init_x,
                                state.drone.y - state._start_position['y'] + drone_init_y)
                        drone_dist_error = drone.compute_distance(belief, truth)

                        if VERBOSE_FLAG:
                            print("Current Belief:", belief)
                            print("True Position:", truth)
                            print("Error:", drone_dist_error, "\n")
           
                    else:
                        display.tree_estimate_at_location(locid,
                                                         x + drone_init_x,
                                                         y + drone_init_y)

            display.end_time_step()

        # One more time step to show final state.
        if VERBOSE_FLAG:
                print('Final State:')
                print(state)

        display.start_time_step()

        display.drone_at_location(state.drone.x, state.drone.y, state.drone.bearing)
        display.end_time_step()

    except Exception as e:
        print(e)

    turtle.bye()


def main(part: str, case: int):
    try:
        if part == 'A':
            part_A(test_cases.IndianaDronesPartATestCases.all_cases[case - 1])
        elif part == 'B':
            part_B(test_cases.IndianaDronesPartBTestCases.all_cases[case - 1])
        else:
            raise ValueError(f'Testing Part {part} is not supported')
    except KeyError:
        raise ValueError(f'Testing Part {part} does not have a case {case}.')


def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument('--part',
                      help="test part",
                      type=str,
                      choices=('A', 'B'))

    prsr.add_argument('--case',
                      help="test case",
                      type=int,
                      default=1,
                      choices=(1, 2, 3, 4, 5))
    return prsr


if __name__ == '__main__':
    args = parser().parse_args()
    main(part=args.part,
         case=args.case)
