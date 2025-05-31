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
# PLAYED AROUND WITH THEESE SO MUCH
AU = 1.49597870700e11
NUM_PARTICLES = 2900
SIGMA = 5e-7  # Increased from 5e-8 to be more lenient with measurements
FUZZ_PERCENTAGE = 0.15  # Fuzz 15% of particles
INITIAL_FUZZ = 1 * AU  # Doubled from 0.002 to 0.004
MIN_FUZZ = 0.001 * AU  # Doubled from 0.0002 to 0.0004
FUZZ_DECAY_RATE = 0.98  # Slower decay to maintain exploration longer
RESAMPLE_VARIATION = 0.001  # Doubled from 0.01 to 0.02

# I consulted LLMs for math and theoretical help
# namely claude 3.5 sonnet
# All work is my own, no code was copy and pasted and no code was taken from other sources such as stack overflow or github

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
        # Randomizing the sun didnt help
        # Sun position within +/- 0.1 AU
        # sun_x = random.uniform(-0.1*AU, 0.1*AU)
        # sun_y = random.uniform(-0.1*AU, 0.1*AU)
        sun_x = 0  # Hard code sun at origin
        sun_y = 0
        
        # originate the particles in a circle around the sun
        for _ in range(NUM_PARTICLES):
            angle = random.uniform(0, 2*pi)
            radius = random.uniform(0.1*AU, 4.0*AU)
            x = sun_x + radius * cos(angle)
            y = sun_y + radius * sin(angle)
            heading = angle + pi/2  # Perpendicular to radius for circular orbit
            particles.append({'x': x, 'y': y, 'heading': heading, 'weight': 1.0})
        other = {'particles': particles, 'sun_x': sun_x, 'sun_y': sun_y, 'time_step': 0}

    particles = other['particles']
    sun_x = other['sun_x']
    sun_y = other['sun_y']
    other['time_step'] += 1
    
    # Calculate current fuzz amount based on time
    # were gonna decay the fuzz over time
    current_fuzz = max(MIN_FUZZ, INITIAL_FUZZ * (FUZZ_DECAY_RATE ** other['time_step']))

    # 1. Calculate weights based on gravity measurement BEFORE moving particles
    # THIS IS CRUCIAL
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

    # 2. Move particles in circular orbits
    # Calculate current best estimate for movement
    x_estimate = sum(p['x'] * p['weight'] for p in particles)
    y_estimate = sum(p['y'] * p['weight'] for p in particles)
    
    # Predict where target will be in next step
    dx = x_estimate - sun_x
    dy = y_estimate - sun_y
    current_radius = sqrt(dx*dx + dy*dy)
    current_angle = atan2(dy, dx)
    
    # Move estimate forward in orbit by one step
    # Big to get that orbital motion
    angle_change = distance / current_radius
    next_angle = current_angle + angle_change + steering
    x_estimate = sun_x + current_radius * cos(next_angle)
    y_estimate = sun_y + current_radius * sin(next_angle)
    
    # Sort particles by weight to find best
    sorted_particles = sorted(particles, key=lambda p: p['weight'], reverse=True)
    num_best = int(NUM_PARTICLES * 0.10)  # Keep top 10% unchanged
    best_particles = sorted_particles[:num_best]
    
    # Move all particles
    for p in particles:
        # Calculate current position relative to sun
        dx = p['x'] - sun_x
        dy = p['y'] - sun_y
        current_radius = sqrt(dx*dx + dy*dy)
        current_angle = atan2(dy, dx)
        
        # Move in circular orbit - exactly matching target's movement
        angle_change = distance / current_radius  # This is the angular velocity
        new_angle = current_angle + angle_change + steering
        
        # Update position
        p['x'] = sun_x + current_radius * cos(new_angle)
        p['y'] = sun_y + current_radius * sin(new_angle)
        p['heading'] = new_angle + pi/2  # Perpendicular to radius

    # 3. Resample particles with improved strategy
    new_particles = particles.copy()  # Start with all existing particles
    
    # Sort particles by weight
    sorted_particles = sorted(particles, key=lambda p: p['weight'], reverse=True)
    
    # Calculate removal and addition numbers while ensuring minimum 300 particles
    # Played with these numbers a bunch, we want to lower it so we have less but still leave some
    num_to_remove = min(int(NUM_PARTICLES * 0.15), len(particles) - 500)  # Remove 15% but keep at least 300
    num_to_add = int(NUM_PARTICLES * 0.13)  # Add 14% new particles
    worst_particles = sorted_particles[-num_to_remove:]
    best_particles = sorted_particles[:num_to_add]  # Use top 14% as source
    
    # Remove worst particles
    for worst_particle in worst_particles:
        new_particles.remove(worst_particle)
    
    # Add new particles from best particles (which have already been moved)
    for i in range(num_to_add):
        # Get corresponding best particle (which is already in its new position)
        chosen = best_particles[i]
        
        # Add variation for resampling from best particles using tunable constant
        # Played with this on and off a bunch
        angle_variation = random.gauss(0, RESAMPLE_VARIATION)  # Angular variation (radians)
        radius_variation = random.gauss(0, AU * RESAMPLE_VARIATION/2)  # Radius variation
        heading_variation = random.gauss(0, RESAMPLE_VARIATION * 2)  # Heading variation (radians)
        
        # Calculate new position with variations from the already-moved position
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
    # Adding more fuzz for sure helped performance
    num_to_fuzz = int(NUM_PARTICLES * FUZZ_PERCENTAGE)
    for _ in range(num_to_fuzz):
        idx = random.randint(0, len(new_particles)-1)  # Note: len might be different now
        
        # Calculate current position relative to sun
        dx = new_particles[idx]['x'] - sun_x
        dy = new_particles[idx]['y'] - sun_y
        current_radius = sqrt(dx*dx + dy*dy)
        current_angle = atan2(dy, dx)
        
        # Fuzz both x and y
        new_particles[idx]['x'] += random.gauss(0, current_fuzz)
        new_particles[idx]['y'] += random.gauss(0, current_fuzz)
        
        # Recalculate heading based on new position
        dx = new_particles[idx]['x'] - sun_x
        dy = new_particles[idx]['y'] - sun_y
        new_angle = atan2(dy, dx)
        new_particles[idx]['heading'] = new_angle + pi/2  # Perpendicular to radius

    # 5. Calculate final estimate
    # Get our actual estimate here, thought this was wrong and it was right the whole time
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

    # other dictionary was very helpful
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

    # This was a crucial part of the process
    if other is None:
        # Initialize with a single position estimate
        angle = random.uniform(0, 2*pi)
        radius = random.uniform(0.1*AU, 4.0*AU)
        x = radius * cos(angle)
        y = radius * sin(angle)
        heading = angle + pi/2  # Perpendicular to radius for circular orbit
        other = {'x': x, 'y': y, 'heading': heading, 'time_step': 0}

    # Get current position
    x = other['x']
    y = other['y']
    heading = other['heading']
    other['time_step'] += 1

    # 1. Move in circular orbit
    # Not sure all of this was necessary but it got me there
    dx = x
    dy = y
    current_radius = sqrt(dx*dx + dy*dy)
    current_angle = atan2(dy, dx)
    
    # Move in circular orbit
    angle_change = distance / current_radius
    new_angle = current_angle + angle_change + steering
    
    # Update position
    x = current_radius * cos(new_angle)
    y = current_radius * sin(new_angle)
    heading = new_angle + pi/2  # Perpendicular to radius

    # 2. Update position based on illumination measurements
    # Try small variations around current position to find better match
    best_x, best_y = x, y
    best_error = float('inf')
    search_radius = 0.1 * AU  # Search within 0.1 AU of current position
    
    # This is just kind of standard trial and error and iteration based on the info from the other planets
    for _ in range(10):  # Try 10 different positions
        test_x = x + random.uniform(-search_radius, search_radius)
        test_y = y + random.uniform(-search_radius, search_radius)
        
        # Get theoretical illumination at test position
        theoretical_illumination = percent_illuminated_sense_func(test_x, test_y)
        
        # Calculate error
        # this is hte meat and potatoes
        error = sum((measured - theoretical)**2 for measured, theoretical 
                   in zip(percent_illuminated_measurements, theoretical_illumination))
        
        if error < best_error:
            best_error = error
            best_x, best_y = test_x, test_y

    # Update position to best match
    x, y = best_x, best_y

    # Calculate bearing to home planet (last planet)
    home_planet = solar_system.planets[-1]
    dx = home_planet.r[0] - x
    dy = home_planet.r[1] - y
    bearing = atan2(dy, dx)

    # Update position in other
    other['x'] = x
    other['y'] = y
    other['heading'] = heading

    # Return points to plot (just the single estimate)
    # could have sent more here but this worked
    optional_points_to_plot = [(x, y, heading)]

    return bearing, (x, y), other, optional_points_to_plot


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
