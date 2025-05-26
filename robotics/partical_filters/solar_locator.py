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

# These import statements give you access to library functions which you may
# (or may not?) want to use.
import random
import time
from math import *
from body import *
from solar_system import *
from satellite import *

# Constants
AU = 1.49597870700e11
NUM_PARTICLES = 1000
SIGMA = 1e-8  # Much smaller sigma for gravitational measurements
FUZZ_PERCENTAGE = 0.3  # More particles to fuzz
FUZZ_AMOUNT = 0.0001 * AU  # Much smaller fuzz amount

def estimate_next_pos(gravimeter_measurement, get_theoretical_gravitational_force_at_point, distance, steering, other=None):
    """
    Estimate the next (x,y) position of the satelite.
    This is the function you will have to write for part A.
    :param gravimeter_measurement: float
        A floating point number representing
        the measured magnitude of the gravitation pull of all the planets
        felt at the target satellite at that point in time.
    :param get_theoretical_gravitational_force_at_point: Func
        A function that takes in (x,y) and outputs a float representing the magnitude of the gravitation pull from
        of all the planets at that (x,y) location at that point in time.
    :param distance: float
        The target satellite's motion distance
    :param steering: float
        The target satellite's motion steering
    :param other: any
        This is initially None, but if you return an OTHER from
        this function call, it will be passed back to you the next time it is
        called, so that you can use it to keep track of important information
        over time. (We suggest you use a dictionary so that you can store as many
        different named values as you want.)
    :return:
        estimate: Tuple[float, float]. The (x,y) estimate of the target satellite at the next timestep
        other: any. Any additional information you'd like to pass between invocations of this function
        optional_points_to_plot: List[Tuple[float, float, float]].
            A list of tuples like (x,y,h) to plot for the visualization
    """
    # time.sleep(1)  # uncomment to pause for the specified seconds each timestep

    # example of how to get the gravity magnitude at a point in the solar system:
    gravity_magnitude = get_theoretical_gravitational_force_at_point(-1*AU, 1*AU)

    # You may optionally also return a list of (x,y,h) points that you would like
    # the PLOT_PARTICLES=True visualizer to plot for visualization purposes.
    # If you include an optional third value, it will be plotted as the heading
    # of your particle.
    # optional_points_to_plot = [(1*AU, 1*AU), (2*AU, 2*AU), (3*AU, 3*AU)]  # Sample (x,y) to plot
    # optional_points_to_plot = [(1*AU, 1*AU, 0.5), (2*AU, 2*AU, 1.8), (3*AU, 3*AU, 3.2)]  # (x,y,heading)

    if other is None:
        # Initialize particles in a tighter region
        particles = []
        for _ in range(NUM_PARTICLES):
            # Random initial position within +/- 1 AU
            x = random.uniform(-1*AU, 1*AU)
            y = random.uniform(-1*AU, 1*AU)
            # Random initial heading
            heading = random.uniform(0, 2*pi)
            particles.append({'x': x, 'y': y, 'heading': heading, 'weight': 1.0})
        other = {'particles': particles}

    particles = other['particles']
    
    # 1. Move particles according to bicycle motion model
    for p in particles:
        # Update heading
        p['heading'] += steering
        # Update position
        p['x'] += distance * cos(p['heading'])
        p['y'] += distance * sin(p['heading'])

    # 2. Calculate weights based on measurement
    total_weight = 0
    for p in particles:
        # Get theoretical gravity at particle position
        theoretical_gravity = get_theoretical_gravitational_force_at_point(p['x'], p['y'])
        # Calculate weight using Gaussian probability
        error = gravimeter_measurement - theoretical_gravity
        # Use log scale for better numerical stability
        p['weight'] = exp(-(abs(error))/(2*SIGMA**2))
        total_weight += p['weight']

    # Normalize weights
    if total_weight > 0:
        for p in particles:
            p['weight'] /= total_weight
    else:
        # If all weights are zero, give equal weight to all particles
        for p in particles:
            p['weight'] = 1.0 / NUM_PARTICLES

    # 3. Resample particles
    new_particles = []
    for _ in range(NUM_PARTICLES):
        # Select particle based on weight
        r = random.random()
        cumsum = 0
        for p in particles:
            cumsum += p['weight']
            if cumsum >= r:
                new_particles.append({
                    'x': p['x'],
                    'y': p['y'],
                    'heading': p['heading'],
                    'weight': 1.0
                })
                break
        # If no particle was selected (shouldn't happen with normalized weights)
        if len(new_particles) < _ + 1:
            new_particles.append({
                'x': random.uniform(-1*AU, 1*AU),
                'y': random.uniform(-1*AU, 1*AU),
                'heading': random.uniform(0, 2*pi),
                'weight': 1.0
            })

    # 4. Fuzz some particles
    num_to_fuzz = int(NUM_PARTICLES * FUZZ_PERCENTAGE)
    for _ in range(num_to_fuzz):
        idx = random.randint(0, len(new_particles)-1)
        # Much smaller fuzz amount
        new_particles[idx]['x'] += random.gauss(0, FUZZ_AMOUNT)
        new_particles[idx]['y'] += random.gauss(0, FUZZ_AMOUNT)
        new_particles[idx]['heading'] += random.gauss(0, 0.001)  # Much smaller heading fuzz

    # 5. Calculate estimate (weighted average)
    x_estimate = sum(p['x'] * p['weight'] for p in new_particles)
    y_estimate = sum(p['y'] * p['weight'] for p in new_particles)
    xy_estimate = (x_estimate, y_estimate)

    # Update particles in other
    other['particles'] = new_particles

    # Optional points to plot (particles)
    optional_points_to_plot = [(p['x'], p['y'], p['heading']) for p in new_particles]

    return xy_estimate, other, optional_points_to_plot


def next_angle(solar_system, percent_illuminated_measurements, percent_illuminated_sense_func,
               distance, steering, other=None):
    """
    Gets the next angle at which to send out an sos message to the home planet,
    the last planet in the solar system.
    This is the function you will have to write for part B.
    :param solar_system: SolarSystem
        A model of the solar system containing the sun and planets as Bodys (contains positions, velocities, and masses)
        Planets are listed in order from closest to furthest from the sun
    :param percent_illuminated_measurements: List[float]
        A list of floating point number from 0 to 100 representing
        the measured percent illumination of each planet in order from closest to furthest to sun
        as seen by the target satellite.
    :param percent_illuminated_sense_func: Func
        A function that takes in (x,y) and outputs the list of percent illuminated measurements of each planet
        as would be seen by satellite at that (x,y) location.
    :param distance: float
        The target satellite's motion distance
    :param steering: float
        The target satellite's motion steering
    :param other: any
        This is initially None, but if you return an OTHER from
        this function call, it will be passed back to you the next time it is
        called, so that you can use it to keep track of important information
        over time. (We suggest you use a dictionary so that you can store as many
        different named values as you want.)
    :return:
        bearing: float. The absolute angle from the satellite to send an sos message between -pi and pi
        xy_estimate: Tuple[float, float]. The (x,y) estimate of the target satellite at the next timestep
        other: any. Any additional information you'd like to pass between invocations of this function
        optional_points_to_plot: List[Tuple[float, float, float]].
            A list of tuples like (x,y,h) to plot for the visualization
    """

    if other is None:
        # Initialize particles
        particles = []
        for _ in range(NUM_PARTICLES):
            x = random.uniform(-4*AU, 4*AU)
            y = random.uniform(-4*AU, 4*AU)
            heading = random.uniform(0, 2*pi)
            particles.append({'x': x, 'y': y, 'heading': heading, 'weight': 1.0})
        other = {'particles': particles}

    particles = other['particles']
    
    # 1. Move particles
    for p in particles:
        p['heading'] += steering
        p['x'] += distance * cos(p['heading'])
        p['y'] += distance * sin(p['heading'])

    # 2. Calculate weights based on percent illumination measurements
    total_weight = 0
    for p in particles:
        # Get theoretical illumination at particle position
        theoretical_illumination = percent_illuminated_sense_func(p['x'], p['y'])
        # Calculate weight using product of Gaussian probabilities
        weight = 1.0
        for measured, theoretical in zip(percent_illuminated_measurements, theoretical_illumination):
            error = measured - theoretical
            weight *= exp(-(error**2)/(2*SIGMA**2))
        p['weight'] = weight
        total_weight += weight

    # Normalize weights
    if total_weight > 0:
        for p in particles:
            p['weight'] /= total_weight

    # 3. Resample particles
    new_particles = []
    for _ in range(NUM_PARTICLES):
        r = random.random()
        cumsum = 0
        for p in particles:
            cumsum += p['weight']
            if cumsum >= r:
                new_particles.append({
                    'x': p['x'],
                    'y': p['y'],
                    'heading': p['heading'],
                    'weight': 1.0
                })
                break

    # 4. Fuzz some particles
    num_to_fuzz = int(NUM_PARTICLES * FUZZ_PERCENTAGE)
    for _ in range(num_to_fuzz):
        idx = random.randint(0, len(new_particles)-1)
        new_particles[idx]['x'] += random.gauss(0, FUZZ_AMOUNT)
        new_particles[idx]['y'] += random.gauss(0, FUZZ_AMOUNT)
        new_particles[idx]['heading'] += random.gauss(0, 0.1)

    # 5. Calculate estimate
    x_estimate = sum(p['x'] * p['weight'] for p in new_particles)
    y_estimate = sum(p['y'] * p['weight'] for p in new_particles)
    xy_estimate = (x_estimate, y_estimate)

    # Calculate bearing to home planet (last planet)
    home_planet = solar_system.planets[-1]
    dx = home_planet.r[0] - x_estimate
    dy = home_planet.r[1] - y_estimate
    bearing = atan2(dy, dx)

    # Update particles in other
    other['particles'] = new_particles

    # You may optionally also return a list of (x,y) or (x,y,h) points that
    # you would like the PLOT_PARTICLES=True visualizer to plot.
    # optional_points_to_plot = [ (1*AU,1*AU), (2*AU,2*AU), (3*AU,3*AU) ]  # Sample plot points
    # Optional points to plot
    # optional_points_to_plot = [(p['x'], p['y'], p['heading']) for p in new_particles]
    optional_points_to_plot = [(p['x'], p['y'], p['heading']) for p in new_particles]

    return bearing, xy_estimate, other, optional_points_to_plot


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
