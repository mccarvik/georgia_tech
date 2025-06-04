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

# I consulted LLMs for math and theoretical help (mostly twiddle algorithm)
    # namely claude 3.5 sonnet
    # All work is my own, no code was copy and pasted and no code was taken from other sources such as stack overflow or github

# Lets submit now and see what happens

def pid_thrust(target_elevation, drone_elevation, tau_p=0, tau_d=0, tau_i=0, data: dict() = {}):
    '''
    Student code for Thrust PID control. Drone's starting x, y position is (0, 0).

    Args:
    target_elevation: The target elevation that the drone has to achieve
    drone_elevation: The drone's elevation at the current time step
    tau_p: Proportional gain
    tau_i: Integral gain
    tau_d: Differential gain
    data: Dictionary that you can use to pass values across calls.
        Reserved keys:
            max_rpm_reached: (True|False) - Whether Drone has reached max RPM in both its rotors.

    Returns:
        Tuple of thrust, data
        thrust - The calculated change in thrust using PID controller
        data - A dictionary containing any values you want to pass to the next
            iteration of this function call.
            Reserved keys:
                max_rpm_reached: (True|False) - Whether Drone has reached max RPM in both its rotors.
    '''
    # Initialize data dictionary if empty
    if not data:
        data = {
            'prev_error': 0,
            'integral': 0,
            'max_rpm_reached': False
        }
    
    # Calculate error
    error = target_elevation - drone_elevation
    
    # Calculate integral term
    data['integral'] += error
    
    # Calculate derivative term
    derivative = error - data['prev_error']
    
    # Calculate PID control output
    thrust = (tau_p * error) + (tau_i * data['integral']) + (tau_d * derivative)
    
    # Store current error for next iteration
    data['prev_error'] = error
    return thrust, data


def pid_roll(target_x, drone_x, tau_p=0, tau_d=0, tau_i=0, data:dict() = {}):
    '''
    Student code for PD control for roll. Drone's starting x,y position is 0, 0.

    Args:
    target_x: The target horizontal displacement that the drone has to achieve
    drone_x: The drone's x position at this time step
    tau_p: Proportional gain, supplied by the test suite
    tau_i: Integral gain, supplied by the test suite
    tau_d: Differential gain, supplied by the test suite
    data: Dictionary that you can use to pass values across calls.

    Returns:
        Tuple of roll, data
        roll - The calculated change in roll using PID controller
        data - A dictionary containing any values you want to pass to the next
            iteration of this function call.

    '''
    # Very similar to the roll function, just with a different error calculation
    # Initialize data dictionary if empty
    if not data:
        data = {
            'prev_error': 0,
            'integral': 0
        }
    
    # Calculate error
    error = target_x - drone_x
    
    # Calculate integral term
    data['integral'] += error
    
    # Calculate derivative term
    # played around with this a lot, 1/10 for now
    derivative = (error - data['prev_error']) / (1/10)
    
    # Calculate PID control output
    roll = (tau_p * error) + (tau_i * data['integral']) + (tau_d * derivative)
    
    # Store current error for next iteration
    data['prev_error'] = error
    return roll, data


def find_parameters_thrust(run_callback, tune='thrust', DEBUG=False, VISUALIZE=False):
    '''
    Student implementation of twiddle algorithm will go here. Here you can focus on
    tuning gain values for Thrust test cases only.

    Args:
    run_callback: A handle to DroneSimulator.run() method. You should call it with your
                PID gain values that you want to test with. It returns an error value that indicates
                how well your PID gain values followed the specified path.

    tune: This will be passed by the test harness.
            A value of 'thrust' means you only need to tune gain values for thrust.
            A value of 'both' means you need to tune gain values for both thrust and roll.

    DEBUG: Whether or not to output debugging statements during twiddle runs
    VISUALIZE: Whether or not to output visualizations during twiddle runs

    Returns:
        tuple of the thrust_params, roll_params:
            thrust_params: A dict of gain values for the thrust PID controller
              thrust_params = {'tau_p': 0.0, 'tau_d': 0.0, 'tau_i': 0.0}

            roll_params: A dict of gain values for the roll PID controller
              roll_params   = {'tau_p': 0.0, 'tau_d': 0.0, 'tau_i': 0.0}

    '''
    # Initialize parameters and their deltas
    params = [0.0, 0.0, 0.0]  # [tau_p, tau_d, tau_i]
    dp = [1.0, 1.0, 1.0]      # Initial step sizes
    
    # Create initial parameter dictionaries
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
    roll_params = {'tau_p': 0, 'tau_d': 0, 'tau_i': 0}
    
    # Get initial error
    hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
    best_error = hover_error
    
    # Twiddle algorithm
    # Took me forever to get this here, not sure it works haha
    tolerance = 0.0001
    while sum(dp) > tolerance:
        for i in range(len(params)):
            # Try increasing parameter
            params[i] += dp[i]
            thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
            # not sure we need the maxes but they are there just in case
            hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
            
            if hover_error < best_error:
                best_error = hover_error
                dp[i] *= 1.1
            else:
                # Try decreasing parameter
                params[i] -= 2 * dp[i]
                thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
                # same here re maxes
                hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
                
                if hover_error < best_error:
                    best_error = hover_error
                    dp[i] *= 1.1
                else:
                    # If neither direction improved, reduce step size
                    params[i] += dp[i]
                    dp[i] *= 0.9
            
            if DEBUG:
                print(f"Parameters: {params}")
                print(f"Best error: {best_error}")
                print(f"Step sizes: {dp}")
    
    # Set final parameters
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
    return thrust_params, roll_params


def find_parameters_with_int(run_callback, tune='thrust', DEBUG=False, VISUALIZE=False):
    '''
    Student implementation of twiddle algorithm will go here. Here you can focus on
    tuning gain values for Thrust test case with Integral error

    Args:
    run_callback: A handle to DroneSimulator.run() method. You should call it with your
                PID gain values that you want to test with. It returns an error value that indicates
                how well your PID gain values followed the specified path.

    tune: This will be passed by the test harness.
            A value of 'thrust' means you only need to tune gain values for thrust.
            A value of 'both' means you need to tune gain values for both thrust and roll.

    DEBUG: Whether or not to output debugging statements during twiddle runs
    VISUALIZE: Whether or not to output visualizations during twiddle runs

    Returns:
        tuple of the thrust_params, roll_params:
            thrust_params: A dict of gain values for the thrust PID controller
              thrust_params = {'tau_p': 0.0, 'tau_d': 0.0, 'tau_i': 0.0}

            roll_params: A dict of gain values for the roll PID controller
              roll_params   = {'tau_p': 0.0, 'tau_d': 0.0, 'tau_i': 0.0}

    '''
    # Initialize parameters and their deltas
    # Start with smaller initial values since we're using integral control
    params = [0.0, 0.0, 0.0]  # [tau_p, tau_d, tau_i]
    dp = [0.1, 0.1, 0.1]      # Smaller initial step sizes for more precise tuning
    
    # Create initial parameter dictionaries
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
    roll_params = {'tau_p': 0, 'tau_d': 0, 'tau_i': 0}
    
    # Get initial error
    hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
    best_error = hover_error
    
    # Twiddle algorithm with integral control considerations
    # Same as above mostly
    tolerance = 0.0001
    while sum(dp) > tolerance:
        for i in range(len(params)):
            # Try increasing parameter
            params[i] += dp[i]
            thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
            hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
            
            # For integral control, we need to be more careful with parameter adjustments
            if hover_error < best_error:
                best_error = hover_error
                # More conservative step size increase for integral term
                if i == 2:  # tau_i
                    dp[i] *= 1.05
                else:
                    dp[i] *= 1.1
            else:
                # Try decreasing parameter
                params[i] -= 2 * dp[i]
                thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
                # again maxes might not be necessary but whatever, the algo I was reading about had this so cant hurt
                hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
                
                if hover_error < best_error:
                    best_error = hover_error
                    # More conservative step size increase for integral term
                    if i == 2:  # tau_i
                        dp[i] *= 1.05
                    else:
                        dp[i] *= 1.1
                else:
                    # If neither direction improved, reduce step size
                    params[i] += dp[i]
                    # More aggressive reduction for integral term
                    if i == 2:  # tau_i
                        dp[i] *= 0.8
                    else:
                        dp[i] *= 0.9
            
            if DEBUG:
                print(f"Parameters: {params}")
                print(f"Best error: {best_error}")
                print(f"Step sizes: {dp}")
    
    # Set final parameters
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
    return thrust_params, roll_params


def find_parameters_with_roll(run_callback, tune='both', DEBUG=False, VISUALIZE=False):
    '''
    Student implementation of twiddle algorithm will go here. Here you will
    find gain values for Thrust as well as Roll PID controllers.

    Args:
    run_callback: A handle to DroneSimulator.run() method. You should call it with your
                PID gain values that you want to test with. It returns an error value that indicates
                how well your PID gain values followed the specified path.

    tune: This will be passed by the test harness.
            A value of 'thrust' means you only need to tune gain values for thrust.
            A value of 'both' means you need to tune gain values for both thrust and roll.

    DEBUG: Whether or not to output debugging statements during twiddle runs
    VISUALIZE: Whether or not to output visualizations during twiddle runs

    Returns:
        tuple of the thrust_params, roll_params:
            thrust_params: A dict of gain values for the thrust PID controller
              thrust_params = {'tau_p': 0.0, 'tau_d': 0.0, 'tau_i': 0.0}

            roll_params: A dict of gain values for the roll PID controller
              roll_params   = {'tau_p': 0.0, 'tau_d': 0.0, 'tau_i': 0.0}

    '''
    # Initialize parameters and their deltas for both thrust and roll
    # [thrust_p, thrust_d, thrust_i, roll_p, roll_d, roll_i]
    params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dp = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Start with smaller steps for both
    
    # Create initial parameter dictionaries
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
    roll_params = {'tau_p': params[3], 'tau_d': params[4], 'tau_i': params[5]}
    
    # Get initial error
    hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
    best_error = hover_error
    
    # Twiddle algorithm for both thrust and roll
    # a little more complicated than above but not much
    # now we got thrust and rolls
    tolerance = 0.0001
    while sum(dp) > tolerance:
        for i in range(len(params)):
            # Try increasing parameter
            params[i] += dp[i]
            # Update both parameter dictionaries
            thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
            roll_params = {'tau_p': params[3], 'tau_d': params[4], 'tau_i': params[5]}
            # again still not using the maxes but they are there if we need them later
            hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
            
            if hover_error < best_error:
                best_error = hover_error
                # More conservative step size increase for integral terms
                if i == 2 or i == 5:  # thrust_i or roll_i
                    dp[i] *= 1.05
                else:
                    dp[i] *= 1.1
            else:
                # Try decreasing parameter
                params[i] -= 2 * dp[i]
                # Update both parameter dictionaries
                thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
                roll_params = {'tau_p': params[3], 'tau_d': params[4], 'tau_i': params[5]}
                # same here with maxes
                hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
                
                # meat and potatoes of the whole shebang here
                if hover_error < best_error:
                    best_error = hover_error
                    # More conservative step size increase for integral terms
                    if i == 2 or i == 5:  # thrust_i or roll_i
                        dp[i] *= 1.05
                    else:
                        dp[i] *= 1.1
                else:
                    # If neither direction improved, reduce step size
                    params[i] += dp[i]
                    # More aggressive reduction for integral terms
                    if i == 2 or i == 5:  # thrust_i or roll_i
                        dp[i] *= 0.8
                    else:
                        dp[i] *= 0.9
            
            if DEBUG:
                print(f"Parameters: {params}")
                print(f"Best error: {best_error}")
                print(f"Step sizes: {dp}")
    
    # Set final parameters
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
    roll_params = {'tau_p': params[3], 'tau_d': params[4], 'tau_i': params[5]}
    
    return thrust_params, roll_params


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = 'kmccarville3'
    return whoami
