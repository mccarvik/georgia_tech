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

import os
import argparse
import json
import random
from game_objects import Asteroid
from settings import generate_asteroid_settings


# get/import required settings

asteroid_attributes         = generate_asteroid_settings['asteroid_attributes']
FIELD_X_BOUNDS              = generate_asteroid_settings['FIELD_X_BOUNDS']
FIELD_Y_BOUNDS              = generate_asteroid_settings['FIELD_Y_BOUNDS']
IN_BOUNDS                   = generate_asteroid_settings['IN_BOUNDS']


class AsteroidGenerator():
    """Creates a group of Asteroids in random positions and initializes motion."""

    def __init__(self, rng, asteroid_properties, field_xy_bounds=(None, None)):

        """Initialize a generator for a group of Asteroid objects."""
        self.random_state = rng
        self.current_count = 1
        self.asteroid_properties = asteroid_properties
        self.xbounds = field_xy_bounds[0]
        self.ybounds = field_xy_bounds[1]

    def generate_asteroid_coeffs(self, asteroids_motion_data, t, simulation_start=False,):
        """Generate random coefficients for Asteroid motion.
        Inputs:
        Returns:
        """

        unique_id = self.current_count
        random_num = self.random_state.random()
        asteroid_speed_x_min = asteroids_motion_data['asteroid_speed_x_min']
        asteroid_speed_x_max = asteroids_motion_data['asteroid_speed_x_max']
        asteroid_speed_y_min = asteroids_motion_data['asteroid_speed_y_min']
        asteroid_speed_y_max = asteroids_motion_data['asteroid_speed_y_max']
        asteroid_accel_x_min = asteroids_motion_data['asteroid_accel_x_min']
        asteroid_accel_x_max = asteroids_motion_data['asteroid_accel_x_max']
        asteroid_accel_y_min = asteroids_motion_data['asteroid_accel_y_min']
        asteroid_accel_y_max = asteroids_motion_data['asteroid_accel_y_max']
        # simulation_start = asteroids_motion_data['simulation_start']

        if simulation_start:
            coeff = {
                "c_pos_x": self.random_state.uniform(*self.xbounds),
                # "c_pos_x": self.random_state.choice((1,1.15)),
                "c_vel_x": self.random_state.uniform(asteroid_speed_x_min, asteroid_speed_x_max) * self.random_state.choice((-1,1)),
                "c_accel_x": self.random_state.uniform(asteroid_accel_x_min, asteroid_accel_x_max) * self.random_state.choice((-1,1)),
                # "c_pos_y": self.random_state.uniform(*FIELD_Y_BOUNDS),
                "c_pos_y": self.random_state.uniform(*self.ybounds),
                # "c_pos_y": 0.01,
                "c_vel_y": self.random_state.uniform(asteroid_speed_y_min, asteroid_speed_y_max) * self.random_state.choice((-1,1)),
                "c_accel_y": self.random_state.uniform(asteroid_accel_y_min, asteroid_accel_y_max) * self.random_state.choice((-1,0.4)),
                "t_created": t,
                "id": unique_id,
            }

        else:
            if random_num < (1/2):
                # generate asteroid from sides
                side = self.random_state.choice(self.xbounds)
                coeff = {
                    "c_pos_x": side,
                    "c_vel_x": self.random_state.uniform(asteroid_speed_x_min, asteroid_speed_x_max) * (1 if side == self.xbounds[0] else -1),
                    "c_accel_x": self.random_state.uniform(asteroid_accel_x_min, asteroid_accel_x_max) * self.random_state.choice((-1,1)),
                    "c_pos_y": self.random_state.uniform(*self.ybounds),
                    "c_vel_y": self.random_state.uniform(asteroid_speed_y_min, asteroid_speed_y_max) * self.random_state.choice((-1,1)),
                    "c_accel_y": self.random_state.uniform(asteroid_accel_y_min, asteroid_accel_y_max) * self.random_state.choice((-1,0.4)),
                    "t_created": t,
                    "id": unique_id,
                }
            else:
                # generate asteroid from top and bottom
                top_bottom = self.random_state.choice(self.ybounds)
                coeff = {
                    # "c_pos_x"  : self.random_state.uniform(*FIELD_X_BOUNDS),
                    "c_pos_x"  : self.random_state.uniform(*self.xbounds),
                    "c_vel_x"  : self.random_state.uniform(asteroid_speed_x_min, asteroid_speed_x_max) * self.random_state.choice((-1,1)),
                    "c_accel_x": self.random_state.uniform(asteroid_accel_x_min, asteroid_accel_x_max) * self.random_state.choice((-1,1)),
                    # "c_pos_y"  : self.random_state.choice(FIELD_Y_BOUNDS),
                    "c_pos_y"  : top_bottom,
                    "c_vel_y"  : self.random_state.uniform(asteroid_speed_y_min, asteroid_speed_y_max) * (1 if top_bottom == self.ybounds[0] else -1),
                    "c_accel_y": self.random_state.uniform(asteroid_accel_y_min, asteroid_accel_y_max) * self.random_state.choice((-1,0.1)),
                    "t_created": t,
                    "id": unique_id,
                }
        return coeff

    def generate(
            self,
            asteroid_motion_data,
    ):
        """Generate list of asteroid objects.
        Inputs:
        Returns:
        """
        # Create range for asteroid coefficients
        asteroids = []

        # Generate collection of asteroids at time = 0
        for _ in range(asteroid_motion_data['num_asteroids_start']):
            t = 0
            coeffs = self.generate_asteroid_coeffs(
                asteroid_motion_data,
                t,
                simulation_start=True
            )
            self.current_count += 1
            a = Asteroid(coeffs)
            asteroids.append(a)

        # Generate asteroids periodically and Distribute the num_asteroids_per_time using gaussian with std 3
        for t in range(1, asteroid_motion_data['time_limit']):
            if asteroid_motion_data['num_asteroids_per_time']<0:
                if t %15 ==0:
                    num_asteroids_per_time_dist = max(0, int(self.random_state.gauss(-asteroid_motion_data['num_asteroids_per_time'], 0)))
                    for _ in range(num_asteroids_per_time_dist):
                        coeffs = self.generate_asteroid_coeffs(asteroid_motion_data, t, )
                        self.current_count += 1
                        a = Asteroid(coeffs)
                        asteroids.append(a)
            else:
                num_asteroids_per_time_dist = max(0, int(self.random_state.gauss(asteroid_motion_data['num_asteroids_per_time'], 0)))
                for _ in range(num_asteroids_per_time_dist):
                    coeffs = self.generate_asteroid_coeffs(
                        asteroid_motion_data,
                        t,
                    )
                    self.current_count += 1
                    a = Asteroid(coeffs)
                    asteroids.append(a)
        return asteroids

def params(args):
    """Process arguments and set up the environment.
    Inputs: Namespaced arguments from the command line
    Returns: Dictionary with test case settings
    """
    rng = random.Random(args.generation_seed)

    x_field_bounds = args.arena_x_bounds
    x_pos_pct = args.agent_x_min_max
    x_pos = [val * x_field_bounds[1] for val in x_pos_pct]
    x_pos = rng.uniform(*x_pos)
    asteroid_generator = AsteroidGenerator(rng=rng, asteroid_properties=args.asteroid_properties,
                                             field_xy_bounds=(args.arena_x_bounds, args.arena_y_bounds))


    asteroid_motion_data= {
        "time_limit": args.time_limit,
        "asteroid_speed_x_min": args.asteroid_speed_x_min,
        "asteroid_speed_x_max": args.asteroid_speed_x_max,
        "asteroid_speed_y_min": args.asteroid_speed_y_min,
        "asteroid_speed_y_max": args.asteroid_speed_y_max,
        "asteroid_accel_x_min": args.asteroid_accel_x_min,
        "asteroid_accel_x_max": args.asteroid_accel_x_max,
        "asteroid_accel_y_min": args.asteroid_accel_y_min,
        "asteroid_accel_y_max": args.asteroid_accel_y_max,
        "num_asteroids_start": args.num_asteroids_start,
        "num_asteroids_per_time": args.num_asteroids_per_time}


    asteroids = asteroid_generator.generate(
        asteroid_motion_data
    )

    return {
        "agent_xpos": x_pos,
        "agent_x_min_max": args.agent_x_min_max,
        "agent_jump_distance": args.agent_jump_distance,
        "arena_x_bounds": args.arena_x_bounds,
        "arena_y_bounds": args.arena_y_bounds,
        "asteroids": [dict(asteroid.params) for asteroid in asteroids],
        "in_bounds": dict(IN_BOUNDS),
        "asteroid_match_range": args.asteroid_match_range,
        "noise_sigma_x": args.noise_sigma_x,
        "noise_sigma_y": args.noise_sigma_y,
        "time_limit": args.time_limit,
        "_args": vars(args),
    }

def main(args):
    """Set up parameters and run the simulation."""
    p = params(args=args)

    # Save location
    outfile = os.path.join("./cases", args.outfile + ".json")
    p["_args"]["outfile"] = os.path.basename(p["_args"]["outfile"]) + ".json"
    with open(outfile, "w+") as f:
        json.dump(p, f, indent=2)
    print("Created %s" % args.outfile)

def get_arguments():
    """
    Inputs: Takes command line arguments
    Returns: Namespaced Object
    """
    prsr = argparse.ArgumentParser(
        "Generate parameters for a test case and write them to file."
    )
    prsr.add_argument(
        "outfile",
        help="name of file to write (the .json extension will be appended to this name)",
    )
    prsr.add_argument(
        "--asteroid_properties",
        help="Select the asteroid properties amongst the options: ['color', 'size', 'form']",
        nargs='*',
        type=str,
        choices=['color', 'size', 'form'],
        # default='red',
    )
    prsr.add_argument(
        "-axmm", "--agent_x_min_max",
        help="Choose the min and max agent x_initialization percent!. Range 0-1",
        nargs='+',
        type=float,
        # default=[0.25, 0.75],
        default=[0.5, 0.5],
    )

    prsr.add_argument(
        "-ajd", "--agent_jump_distance",
        help="Radius around which an agent can hop/jump to a asteroid",
        type=float,
        default=0.4,
    )
    prsr.add_argument(
        "-axb", "--arena_x_bounds",
        help="Choose the min and max Arena x_bounds",
        nargs='+',
        type=int,
        default=[0, 2],
    )
    prsr.add_argument(
        "-ayb", "--arena_y_bounds",
        help="Choose the min and max Arena x_bounds",
        nargs='+',
        type=int,
        default=[0, 2],
    )

    prsr.add_argument(
        "-xvmax", "--asteroid_speed_x_max",
        help="maximum magnitude for asteroid x velocity term coefficient",
        type=float,
        default=0.03,
    )
    prsr.add_argument(
        "-xvmin", "--asteroid_speed_x_min",
        help="minimum magnitude for asteroid x velocity term coefficient",
        type=float,
        default=0.01,
    )
    prsr.add_argument(
        "-yvmax", "--asteroid_speed_y_max",
        help="maximum magnitude for asteroid y velocity term coefficient",
        type=float,
        default=0.05,
    )
    prsr.add_argument(
        "-yvmin", "--asteroid_speed_y_min",
        help="minimum magnitude for asteroid y velocity term coefficient",
        type=float,
        default=0.01,
    )
    prsr.add_argument(
        "-xamin", "--asteroid_accel_x_min",
        help="minimum magnitude for asteroid x acceleration term coefficient",
        type=float,
        default=0.0,
    )
    prsr.add_argument(
        "-xamax", "--asteroid_accel_x_max",
        help="maximum magnitude for asteroid x acceleration term coefficient",
        type=float,
        default=0.0,
    )
    prsr.add_argument(
        "-yamin", "--asteroid_accel_y_min",
        help="minimum magnitude for asteroid acceleration term coefficient",
        type=float,
        default=0.0,
    )
    prsr.add_argument(
        "-yamax", "--asteroid_accel_y_max",
        help="maximum magnitude for asteroid acceleration term coefficient",
        type=float,
        default=0.0,
    )

    prsr.add_argument(
        "-md", "--asteroid_match_range",
        help="minimum distance estimate must be from asteroid location to be considered correct",
        type=float,
        default=0.03,
    )
    prsr.add_argument(
        "--noise_sigma_x",
        help="sigma of Gaussian noise applied to the x-component of asteroid measurements",
        type=float,
        default=0.045,
    )
    prsr.add_argument(
        "--noise_sigma_y",
        help="sigma of Gaussian noise applied to the y-component of asteroid measurements",
        type=float,
        default=0.075,
    )
    prsr.add_argument(
        "--num_asteroids_per_time",
        help="number of asteroids created each timestep",
        type=int,
        default= 4,
    )
    prsr.add_argument(
        "--num_asteroids_start",
        help="number of asteroids at time = 0",
        type=int,
        default=10,
    )
    prsr.add_argument(
        "--observation_noise_seed",
        help="random seed to use when generating asteroids",
        type=int,
        default=random.randint(0,200),
    )
    prsr.add_argument(
        "--generation_seed",
        help="random seed to use when generating asteroids",
        type=int,
        required=False,
    )
    prsr.add_argument(
        "--time_limit", help="Number of timesteps to simulate", type=int, default=400
    )
    return prsr


if __name__ == "__main__":
    args = get_arguments().parse_args()
    main(args)
