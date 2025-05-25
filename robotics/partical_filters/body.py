from math import *
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

G = 6.6743e-11


class Body:
    """A body of mass.

    Attributes:
        r(List): the position vector of the body as [x, y]
        v(List): the velocity vector of the body as [v_x, v_y]
        mass(float): the mass of the body
    """

    def __init__(self, r, v, mass):
        """
        Initialize a Body
        If no r position vector or v velocity vector is provided, sets these at default value [0, 0]
        :param r: List[float, float]. The position of the body, such as [x, y]
        :param v: List[float, float]. The velocity of the body, such as [vx, vy]
        :param mass: float. The mass of the body
        """
        self.r = r if r is not None else [0, 0]
        self.v = v if v is not None else [0, 0]
        self.mass = mass

    @classmethod
    def create_body_at_xy_in_orbit(cls, r, mass, r_sun, mass_sun):
        """
        Given a position vector and mass of sun, initializes a Body in circular orbit
        :param r: List[float, float]. The position of the body, such as [x, y]
        :param mass: float. The mass of the body
        :param r_sun: List[float, float]. The position of the sun, such as [x, y]
        :param mass_sun: float. The mass of the sun
        """
        dx = r[0] - r_sun[0]
        dy = r[1] - r_sun[1]
        radius_body = sqrt(dx**2 + dy**2)
        angle = atan2(dy, dx)
        velocity_magnitude = sqrt(G * mass_sun / radius_body)  # m/s
        heading = angle + pi / 2  # perpendicular
        velocity_x = velocity_magnitude * cos(heading)
        velocity_y = velocity_magnitude * sin(heading)

        return cls(r, [velocity_x, velocity_y], mass)

    def get_radius(self):
        """ Returns the radius or distance from center """
        return sqrt(self.r[0]**2 + self.r[1]**2)

    def __repr__(self):
        """This allows us to print a Body's position

        Returns:
            String representation of a Body
        """
        return f'(r=[{self.r[0]:.0f}, {self.r[1]:.0f}], v=[{self.v[0]:.0f}, {self.v[1]:.0f}], mass={self.mass:.2f})'
