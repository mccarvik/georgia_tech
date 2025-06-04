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
    tolerance = 0.0001
    while sum(dp) > tolerance:
        for i in range(len(params)):
            # Try increasing parameter
            params[i] += dp[i]
            thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
            hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)
            
            if hover_error < best_error:
                best_error = hover_error
                dp[i] *= 1.1
            else:
                # Try decreasing parameter
                params[i] -= 2 * dp[i]
                thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}
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

    # Initialize a list to contain your gain values that you want to tune, e.g.,
    params = [0,0,0]

    # Create dicts to pass the parameters to run_callback
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}

    # If tuning roll, then also initialize gain values for roll PID controller
    roll_params   = {'tau_p': 0, 'tau_d': 0, 'tau_i': 0}

    # Call run_callback, passing in the dicts of thrust and roll gain values
    hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)

    # Calculate best_error from above returned values
    best_error = None

    # Implement your code to use twiddle to tune the params and find the best_error

    # Return the dict of gain values that give the best error.

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
    # Initialize a list to contain your gain values that you want to tune, e.g.,
    params = [0,0,0]

    # Create dicts to pass the parameters to run_callback
    thrust_params = {'tau_p': params[0], 'tau_d': params[1], 'tau_i': params[2]}

    # If tuning roll, then also initialize gain values for roll PID controller
    roll_params   = {'tau_p': 0, 'tau_d': 0, 'tau_i': 0}

    # Call run_callback, passing in the dicts of thrust and roll gain values
    hover_error, max_allowed_velocity, drone_max_velocity, max_allowed_oscillations, total_oscillations = run_callback(thrust_params, roll_params, VISUALIZE=VISUALIZE)

    # Calculate best_error from above returned values
    best_error = None

    # Implement your code to use twiddle to tune the params and find the best_error

    # Return the dict of gain values that give the best error.

    return thrust_params, roll_params

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith225).
    whoami = ''
    return whoami
