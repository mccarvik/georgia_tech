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
SIGMA = 1e-8  # Sigma for gravitational measurements
FUZZ_PERCENTAGE = 0.15  # Fuzz 20% of particles
FUZZ_AMOUNT = 0.0001 * AU  # Fuzz amount (reduced by 10x)
RESAMPLE_VARIATION = 0.005  # Tunable constant for resampling variation

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
    if other is None:
        # Initialize particles uniformly in plausible region
        particles = []
        # Sun position within +/- 0.1 AU
        sun_x = random.uniform(-0.1*AU, 0.1*AU)
        sun_y = random.uniform(-0.1*AU, 0.1*AU)
        
        for _ in range(NUM_PARTICLES):
            angle = random.uniform(0, 2*pi)
            radius = random.uniform(0.1*AU, 4.0*AU)
            x = sun_x + radius * cos(angle)
            y = sun_y + radius * sin(angle)
            heading = angle + pi/2  # Perpendicular to radius for circular orbit
            particles.append({'x': x, 'y': y, 'heading': heading, 'weight': 1.0})
        other = {'particles': particles, 'sun_x': sun_x, 'sun_y': sun_y}

    particles = other['particles']
    sun_x = other['sun_x']
    sun_y = other['sun_y']

    # 1. Move particles in circular orbits
    # Calculate current best estimate for movement
    x_estimate = sum(p['x'] * p['weight'] for p in particles)
    y_estimate = sum(p['y'] * p['weight'] for p in particles)
    
    # Predict where target will be in two steps ahead
    dx = x_estimate - sun_x
    dy = y_estimate - sun_y
    current_radius = sqrt(dx*dx + dy*dy)
    current_angle = atan2(dy, dx)
    
    # Move estimate forward in orbit by two steps
    angle_change = distance / current_radius
    next_angle = current_angle + (2 * angle_change) + (2 * steering)  # Double the movement
    x_estimate = sun_x + current_radius * cos(next_angle)
    y_estimate = sun_y + current_radius * sin(next_angle)
    
    # Sort particles by weight to find best
    sorted_particles = sorted(particles, key=lambda p: p['weight'], reverse=True)
    num_best = int(NUM_PARTICLES * 0.10)  # Keep top 10% unchanged
    best_particles = sorted_particles[:num_best]
    
    for p in particles:
        # Calculate current position relative to sun
        dx = p['x'] - sun_x
        dy = p['y'] - sun_y
        current_radius = sqrt(dx*dx + dy*dy)
        current_angle = atan2(dy, dx)
        
        # Move in circular orbit - identical to how target moves
        angle_change = distance / current_radius
        new_angle = current_angle + angle_change + steering  # Only use the provided steering
        
        # Update position
        p['x'] = sun_x + current_radius * cos(new_angle)
        p['y'] = sun_y + current_radius * sin(new_angle)
        p['heading'] = new_angle + pi/2  # Perpendicular to radius

    # 2. Calculate weights based on gravity measurement
    total_weight = 0
    for p in particles:
        theoretical_gravity = get_theoretical_gravitational_force_at_point(p['x'], p['y'])
        error = gravimeter_measurement - theoretical_gravity
        
        # Calculate weight based solely on gravity match
        p['weight'] = exp(-((error)**2)/(2*SIGMA**2))
        total_weight += p['weight']

    # Normalize weights
    if total_weight > 0:
        for p in particles:
            p['weight'] /= total_weight
    else:
        for p in particles:
            p['weight'] = 1.0 / NUM_PARTICLES

    # 3. Resample particles with improved strategy
    new_particles = particles.copy()  # Start with all existing particles
    
    # Sort particles by weight to find best and worst
    sorted_particles = sorted(particles, key=lambda p: p['weight'], reverse=True)
    
    # Calculate removal and addition numbers while ensuring minimum 100 particles
    num_to_remove = min(int(NUM_PARTICLES * 0.11), len(particles) - 100)  # Remove 3% but keep at least 100
    num_to_add = int(NUM_PARTICLES * 0.11)  # Add 1% new particles
    worst_particles = sorted_particles[-num_to_remove:]
    best_particles = sorted_particles[:num_to_add]  # Use top 1% as source
    
    # Remove worst particles
    for worst_particle in worst_particles:
        new_particles.remove(worst_particle)
    
    # Add new particles from best particles
    for i in range(num_to_add):
        # Get corresponding best particle
        chosen = best_particles[i]
        
        # Add variation for resampling from best particles using tunable constant
        angle_variation = random.gauss(0, RESAMPLE_VARIATION)  # Angular variation (radians)
        radius_variation = random.gauss(0, AU * RESAMPLE_VARIATION/2)  # Radius variation
        heading_variation = random.gauss(0, RESAMPLE_VARIATION * 2)  # Heading variation (radians)
        
        # Calculate new position with variations
        current_radius = sqrt((chosen['x'] - sun_x)**2 + (chosen['y'] - sun_y)**2)
        current_angle = atan2(chosen['y'] - sun_y, chosen['x'] - sun_x)
        new_radius = current_radius + radius_variation
        new_angle = current_angle + angle_variation
        
        new_particles.append({
            'x': sun_x + new_radius * cos(new_angle),
            'y': sun_y + new_radius * sin(new_angle),
            'heading': chosen['heading'] + angle_variation + heading_variation,
            'weight': 1.0
        })

    # 4. Fuzz a small percentage of particles
    num_to_fuzz = int(NUM_PARTICLES * FUZZ_PERCENTAGE)
    for _ in range(num_to_fuzz):
        idx = random.randint(0, len(new_particles)-1)  # Note: len might be different now
        # Calculate current radius and angle
        dx = new_particles[idx]['x'] - sun_x
        dy = new_particles[idx]['y'] - sun_y
        current_radius = sqrt(dx*dx + dy*dy)
        current_angle = atan2(dy, dx)
        
        # Calculate distance to target estimate
        dx_target = x_estimate - new_particles[idx]['x']
        dy_target = y_estimate - new_particles[idx]['y']
        dist_to_target = sqrt(dx_target*dx_target + dy_target*dy_target)
        
        # Fuzz angle while steering towards target
        new_angle = current_angle + random.gauss(0, FUZZ_AMOUNT/current_radius)  # Use FUZZ_AMOUNT constant
        target_steering = 0.1 * (dist_to_target / AU)  # Reduced steering for fuzzed particles
        
        new_particles[idx]['x'] = sun_x + current_radius * cos(new_angle)
        new_particles[idx]['y'] = sun_y + current_radius * sin(new_angle)
        new_particles[idx]['heading'] = new_angle + pi/2 + target_steering

    # 5. Calculate estimate (weighted average)
    # Recalculate weights for estimate
    total_weight = 0
    for p in new_particles:
        theoretical_gravity = get_theoretical_gravitational_force_at_point(p['x'], p['y'])
        error = gravimeter_measurement - theoretical_gravity
        p['weight'] = exp(-((error)**2)/(2*SIGMA**2))
        total_weight += p['weight']
    if total_weight > 0:
        for p in new_particles:
            p['weight'] /= total_weight
    else:
        for p in new_particles:
            p['weight'] = 1.0 / NUM_PARTICLES
    x_estimate = sum(p['x'] * p['weight'] for p in new_particles)
    y_estimate = sum(p['y'] * p['weight'] for p in new_particles)
    xy_estimate = (x_estimate, y_estimate)

    other['particles'] = new_particles
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
