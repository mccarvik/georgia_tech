from math import *
from body import Body

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
AU = 1.49597870700e11


class SolarSystem:
    """A solar system of planets orbiting around a sun"""

    def __init__(self, mass_sun, r_sun,
                 planets_r_and_mass=None,
                 elliptical_orbit_factor=1,
                 ):
        self.sun = Body(r=r_sun, v=[0, 0], mass=mass_sun)
        self.planets = [self.init_body_in_orbit_at_x_and_y(r_body, mass_body, r_sun, mass_sun * elliptical_orbit_factor)
                        for r_body, mass_body in planets_r_and_mass]

    @classmethod
    def init_body_in_orbit_at_radius_and_angle(cls, radius_body, angle_body, mass_body, r_sun, mass_sun):
        """
        Initializes a body at the given radius and angle
        with position vector r and velocity vector v
        such that the body is in orbit around the sun
        :param radius_body: float. The radius defined as distance from the body to the sun.
        :param angle_body: float. The angle defined by x-axis, sun at vertex, and sun-body.
        :param mass_body: float. The mass of the body
        :param r_sun: List[float, float]. The position of the sun, such as [x, y]
        :param mass_sun: float. The mass of the sun
        """

        x = radius_body * cos(angle_body)
        y = radius_body * sin(angle_body)
        r = [x, y]

        return cls.init_body_in_orbit_at_x_and_y(r_body=r, mass_body=mass_body, r_sun=r_sun, mass_sun=mass_sun, )

    @classmethod
    def init_body_in_orbit_at_x_and_y(cls, r_body, mass_body, r_sun, mass_sun):
        """
        Initializes a body at the given position vector r
        and with velocity vector v such that the body is in orbit around the sun
        :param r_body: List[float, float]. The position of the body, such as [x, y]
        :param mass_body: float. The mass of the body
        :param r_sun: List[float, float]. The position of the sun, such as [x, y]
        :param mass_sun: float. The mass of the sun
        """
        return Body.create_body_at_xy_in_orbit(r_body, mass_body, r_sun, mass_sun)

    def move_planets(self):
        """
        Moves/rotates all planets in the system by one time step in place.
        """
        for planet in self.planets:
            self.move_body(planet)

    def move_body(self, body: Body):
        """
        Moves the provided body one time step in accordance with the gravitational pull of the sun
        """
        dt = 60 * 60 * 1  # 1 hr

        for k in range(24):  # 1 day
            # catch for divide by zero by shifting it 1 meter
            if body.r[0] == self.sun.r:
                body.r[0] += 1
            dx = body.r[0] - self.sun.r[0]
            dy = body.r[1] - self.sun.r[1]
            # acceleration = G * M / r^2 = G * M / |r|^2 * -r / |r|
            c = -G * self.sun.mass / (dx ** 2 + dy ** 2) ** (3. / 2)  # m/s^2
            acc = [c * [dx, dy][i] for i in range(2)]
            body.v = [body.v[i] + (acc[i] * dt) for i in range(2)]
            body.r = [body.r[i] + body.v[i] * dt for i in range(2)]

        return body

    def get_all_bodies(self):
        """ Gets all bodies in the solar system, including the sun """
        return [self.sun] + [body for body in self.planets]

    def get_num_planets(self):
        """ Gets the number of planets in the solar system """
        return len(self.planets)

    def __repr__(self):
        """Prints a string representation of the solar system
        """
        return '[\n  '+',\n  '.join([str(body) for body in self.get_all_bodies()]) + '\n]'
