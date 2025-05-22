import random
from settings import FIELD_X_BOUNDS, FIELD_Y_BOUNDS

class Arena():
    """The bounds in which Aliens will fall during the simulation."""

    def __init__(self, x_bounds=FIELD_X_BOUNDS, y_bounds=FIELD_Y_BOUNDS, agent_xstart_min_max=FIELD_X_BOUNDS):
        """Initialize the Arena."""
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.agent_xstart_min = agent_xstart_min_max[0] * x_bounds[1]
        self.agent_xstart_max = agent_xstart_min_max[1] * x_bounds[1]


    def contains(self, point_xy):
        """Return True if point_xy is within this Arena, false otherwise."""
        return ((self.x_bounds[0] <= point_xy[0] <= self.x_bounds[1])
                and (self.y_bounds[0] <= point_xy[1] <= self.y_bounds[1]))

    def __repr__(self):
        """How the Arena is represented when printed out to the cli."""
        return f'(({self.x_bounds[0]}, {self.y_bounds[0]}), ({self.x_bounds[1]}, {self.y_bounds[1]}))'

    @property
    def bounds(self):
        """Return the bounds of the Arena."""
        return {'x': self.x_bounds,
                'y': self.y_bounds}


class Agent():
    def __init__(self, x_pos=None, y_pos=None, jump_distance=0.2, x_bounds=None, y_bounds=None):
        agent_min_val = x_pos[0]
        if x_pos is None:
            self.x_pos = random.uniform(*x_pos)
        else:
            min_val = x_pos[0]*x_bounds[1]
            max_val = x_pos[1]*x_bounds[1]
            self.x_pos = random.uniform(min_val, max_val)
        # self.y_pos = y_pos if (y_pos is not None) else FIELD_Y_BOUNDS[0]
        self.y_pos = y_bounds[0] if (y_pos is None) else y_pos
        self.x_vel = 0
        self.y_vel = 0
        self.x_acc = 0
        self.y_acc = 0
        self.jump_distance = jump_distance
        self.isriding = None

    def get_agent_position(self, t=0):
        self.x_pos = self.x_pos + (self.x_vel * t) + (0.5 * self.x_acc * t**2)
        self.y_pos = self.y_pos + (self.y_vel * t) + (0.5 * self.y_acc * t**2)
        return (self.x_pos, self.y_pos)

    def set_agent_position(self, x, y):
        self.x_pos = x
        self.y_pos = y


class Asteroid():
    """An asteroid trying to invade Earth."""

    def __init__(self, asteroid_params):
        """Initialize the asteroid and its motion coefficients."""
        self.c_pos_x = asteroid_params['c_pos_x']
        self.c_vel_x = asteroid_params['c_vel_x']
        self.c_accel_x = asteroid_params['c_accel_x']
        self.c_pos_y = asteroid_params['c_pos_y']
        self.c_vel_y = asteroid_params['c_vel_y']
        self.c_accel_y = asteroid_params['c_accel_y']
        self.t_created = asteroid_params['t_created']
        self.id = asteroid_params['id']
        self.color = ['red', 'blue', 'green']
        self.active = True

    def xy_pos(self, t):
        """Return the x-y position of this asteroid."""
        t_shifted = t - self.t_created

        x_pos = ((0.5 * self.c_accel_x * t_shifted * t_shifted)
                 + (self.c_vel_x * t_shifted)
                 + self.c_pos_x)

        y_pos = ((0.5 * self.c_accel_y * t_shifted * t_shifted)
                 + (self.c_vel_y * t_shifted)
                 + self.c_pos_y)
        return (x_pos, y_pos)

    def xy_vel(self, t):
        t_shifted = t - self.t_created

        x_vel = self.c_vel_x + self.c_accel_x * t_shifted
        y_vel = self.c_vel_y + self.c_accel_y * t_shifted
        return (x_vel, y_vel)

    @property
    def params(self):
        """Return parameters of this asteroid."""
        return {'c_pos_x'  : self.c_pos_x,
                'c_vel_x'  : self.c_vel_x,
                'c_accel_x': self.c_accel_x,
                'c_pos_y'  : self.c_pos_y,
                'c_vel_y'  : self.c_vel_y,
                'c_accel_y': self.c_accel_y,
                't_created': self.t_created,
                'id'       : self.id
                }

    def deactivate(self):
        """Deactivate this asteroid."""
        self.id = -1

        # self.active = False
    def __del__(self):
        pass


class AsteroidShower():
    """A collection of Asteroids."""

    def __init__(self, thearena, seed,
                 asteroids, spaceship):
        """Initialize the collection of asteroids."""
        self.asteroids = asteroids
        self.arenacontains = thearena.contains
        self.x_bounds = thearena.bounds['x']
        self.y_bounds = thearena.bounds['y']
        self.spaceship = spaceship
        self.random_state = random.Random(seed)
        self.outside_bounds = {
            'xleft'   : False,
            'xright'  : False,
            'ytop'    : False,
            'ybottom' : False,
            }


    def asteroid_locations(self, time, seen=None):
        """Return the asteroids' locations.

        This returns a list of tuples, each of which contains a specific
        asteroid's index, x-position, and y-position.
        """

        locs = {}
        outside_field_l_r_b = set()
        outside_field = set()
        # seen = set()

        for asteroid in self.asteroids:
            if asteroid.t_created <= time:
                xyloc = asteroid.xy_pos(time)
                xyvel = asteroid.xy_vel(time)
                if self.arenacontains((xyloc[0], xyloc[1])):
                    if asteroid.id not in seen:
                        # locs[asteroid.id] = {'x': xyloc[0], 'y': xyloc[1], 'x_vel':xyvel[0], 'y_vel':xyvel[1], 'x_acc': asteroid.c_accel_x, 'y_acc': asteroid.c_accel_y, 'type':'rock'}
                        # locs[asteroid.id] = {'x': xyloc[0], 'y': xyloc[1]}
                        locs[asteroid.id] = (xyloc[0],  xyloc[1])
                else:
                    if asteroid.xy_pos(time)[1] < self.y_bounds[0] or\
                            asteroid.xy_pos(time)[0] < self.x_bounds[0] or\
                                asteroid.xy_pos(time)[0] > self.x_bounds[1]:
                        outside_field_l_r_b.add(asteroid.id)

                    if asteroid.xy_pos(time)[1] < self.y_bounds[0] or\
                        asteroid.xy_pos(time)[1] > self.y_bounds[1] or\
                            asteroid.xy_pos(time)[0] < self.x_bounds[0] or\
                                asteroid.xy_pos(time)[0] > self.x_bounds[1]:
                        outside_field.add(asteroid.id)

        return locs, outside_field, outside_field_l_r_b


    def deactivate_if_out_of_bounds(self, time):
        """Delete asteroids that move out of bounds."""
        for asteroid in self.asteroids:
            # if asteroid.id < 0:
            #     continue

            prev_xyloc = asteroid.xy_pos(time-1)
            in_bounds_prior_timestep = self.arenacontains((prev_xyloc[0], prev_xyloc[1]))
            out_of_bounds_current_timestep = (asteroid.xy_pos(time)[0] < self.x_bounds[0] or
                                              asteroid.xy_pos(time)[0] > self.x_bounds[1] or
                                              asteroid.xy_pos(time)[1] < self.y_bounds[0] or
                                              asteroid.xy_pos(time)[1] > self.y_bounds[1])
            already_created = asteroid.t_created <= time

            #  identify asteroid as outofbounds and from which boundary that occurred


            # deactivate asteroid if it moves out of bounds
            if in_bounds_prior_timestep and \
                out_of_bounds_current_timestep and \
                already_created:
                # asteroid.deactivate()
                self.asteroids.remove(asteroid)

        return None

