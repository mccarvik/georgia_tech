import random

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


class Satellite:
    """A man-made satellite.

    Attributes:
        x: x position
        y: y position
        h: heading
        g_measurement_noise: gravimeter measurement noise
    """

    def __init__(self, x, y, h, l, mass, g_measurement_noise, percent_illuminated_measurement_noise):
        """

        :param x: The x position of the Satellite.
        :param y: The y position of the Satellite.
        :param h: The heading of the Satellite.
        :param l: The length of the Satellite.
        :param mass: float. The mass of the Satellite.
        :param g_measurement_noise: float. The measurement noise of the gravimeter on the Satellite.
        :param percent_illuminated_measurement_noise: The measurement noise of the device on the Satellite
               taking percent illuminated readings
        """
        self.x = x
        self.y = y
        self.h = h
        self.l = l
        self.mass = mass
        self.g_measurement_noise = g_measurement_noise
        self.percent_illuminated_measurement_noise = percent_illuminated_measurement_noise

    def sense_gravimeter(self, measurement_function):
        """
        Measures the magnitude of the sum of gravitational acceleration vectors
        from the planets at this Satellite.
        """
        measurement = measurement_function(self.x, self.y)
        return random.gauss(measurement, self.g_measurement_noise)

    def sense_percent_illuminated(self, measurement_function):
        """
        Measures the percent illuminated for each planet in the solar system as seen at the satellite.
        """
        measurements = measurement_function(self.x, self.y)
        return [random.gauss(measurement, self.percent_illuminated_measurement_noise) for measurement in measurements]

    def __repr__(self):
        """This allows us to print a Satellite's position

        Returns:
            String representation of a Satellite
        """
        return f'(x,y,h,l)=({self.x:.0f}, {self.y:.0f}, {self.h:.0f}, {self.l:.0f})'
