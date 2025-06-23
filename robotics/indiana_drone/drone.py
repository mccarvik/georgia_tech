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
import random
from typing import Tuple

PI = math.pi


def compute_distance(p: Tuple, q: Tuple):
    """Compute the distance between two points.

    Args:
        p: Point 1
        q: Point 2

    Returns:
        The Euclidean distance between the points.
    """

    x1, y1 = p
    x2, y2 = q

    dx = x2 - x1
    dy = y2 - y1

    return math.sqrt(dx**2 + dy**2)


def compute_bearing(p: Tuple, q: Tuple):
    """
    Compute bearing between two points.

    Args:
        p: Point 1
        q: Point 2

    Returns:
        The bearing as referenced from the horizontal axis.
    """
    x1, y1 = p
    x2, y2 = q

    dx = x2 - x1
    dy = y2 - y1

    return math.atan2(dy, dx)


def truncate_angle(t: float):
    """
    Truncate the angle between -PI and PI

    Args:
        t: angle to truncate.

    Returns:
        Truncated angle.
    """
    return ((t+PI) % (2*PI)) - PI


class Drone:
    """
    Attributes:
        x: x location.
        y: y location.
        bearing: angle from horizontal axis in radians.
        max_distance: max distance drone can move.
        max_steering: max turning angle.
        measure_distance_noise: noise of distance measurement to a point.
        measure_bearing_noise: noise of bearing measurement to a point.
        move_steering_noise: noise in move steering.
        move_distance_noise: noise in move distance.
    """
    def __init__(self, x: float = 0.0, y: float = 0.0, bearing: float = 0.0, max_distance: float = float('inf'),
                 max_steering: float = float('inf'),  move_steering_noise: float = 0.0, move_distance_noise: float = 0.0,
                 measure_distance_noise: float = 0.0, measure_bearing_noise: float = 0.0,
                 sensor_decay_rate: float = 0.01):

        self.x = x
        self.y = y
        self.bearing = bearing
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.measure_distance_noise = measure_distance_noise
        self.measure_bearing_noise = measure_bearing_noise
        self.move_steering_noise = move_steering_noise
        self.move_distance_noise = move_distance_noise
        self.sensor_decay_rate = sensor_decay_rate

    def set_noise(self, move_steering_noise: float, move_distance_noise: float, measure_distance_noise: float,
                  measure_bearing_noise: float):
        """
        Set noise values for this drone.

        Args:
            move_steering_noise: noise in move steering.
            move_distance_noise: noise in move distance.
            measure_distance_noise: noise of distance measurement to a point.
            measure_bearing_noise: noise of bearing measurement to a point.
        """
        self.move_steering_noise = float(move_steering_noise)
        self.move_distance_noise = float(move_distance_noise)
        self.measure_distance_noise = float(measure_distance_noise)
        self.measure_bearing_noise = float(measure_bearing_noise)

    def move(self, distance: float, steering: float, noise: bool = False):
        """
        Move the drone.

        Args:
            steering: Steering angle to turn before move relative to current heading.
            distance: Distance to move.
            noise: Move using set noise values.
        """
        if noise:
            steering += random.gauss(0.0, self.move_steering_noise)
            distance += random.gauss(0.0, self.move_distance_noise)

        steering = max(-self.max_steering, steering)
        steering = min(self.max_steering, steering)
        distance = max(0.0, distance)
        distance = min(self.max_distance, distance)

        self.bearing = truncate_angle(self.bearing + float(steering))
        self.x += distance * math.cos(self.bearing)
        self.y += distance * math.sin(self.bearing)

    def measure_distance_and_bearing_to(self, point: Tuple, noise: bool = False):
        """
        Measure the distance and bearing to a point.

        Args:
            point: Point to take measurement reading to.
            noise: Measure using set noise values.

        Returns:
            The distance and bearing to the point.
        """
        current_position = (self.x, self.y)

        distance_to_point = compute_distance(current_position, point)
        bearing_to_point = compute_bearing(current_position, point)
        bearing_to_point = bearing_to_point - self.bearing

        if noise:
            distance_to_point += random.gauss(0.0, self.measure_distance_noise)
            bearing_to_point += random.gauss(0.0, self.measure_bearing_noise)

        bearing_to_point = truncate_angle(bearing_to_point)

        return distance_to_point, bearing_to_point

    def __repr__(self):
        """
        This allows us to print a drone's position
        """
        return f'[self.x:.5f, self.y:.5f]'
