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

import random
from utilities import distance_formula, clamp
from settings import runner_settings
from collections import defaultdict, Counter
from copy import deepcopy
import statistics as stats

DISPLAY_ESTIMATED_SPACESHIP_MATCH_RANGE  = runner_settings['DISPLAY_ESTIMATED_SPACESHIP_MATCH_RANGE']
DISPLAY_ASTEROID_MATCH_RANGE             = runner_settings['DISPLAY_ASTEROID_MATCH_RANGE']
FAILURE_ASTEROID_OUT_OF_JUMP_RANGE       = runner_settings['FAILURE_ASTEROID_OUT_OF_JUMP_RANGE']
FAILURE_TIME_STEP_LIMIT_EXCEEDED         = runner_settings['FAILURE_TIME_STEP_LIMIT_EXCEEDED']
FAILURE_LOW_SCORE                        = runner_settings['FAILURE_LOW_SCORE']
FAILURE_UNIMPLEMENTED                    = runner_settings['FAILURE_UNIMPLEMENTED']
FAILURE_AGENT_OUT_OF_BOUNDS              = runner_settings['FAILURE_AGENT_OUT_OF_BOUNDS']
DISPLAY_ORIGIN_TO_DESTINATION_LINE       = runner_settings['DISPLAY_ORIGIN_TO_DESTINATION_LINE']
NUM_TIMESTEPS_TO_PREDICT_INFUTURE        = runner_settings['NUM_TIMESTEPS_TO_PREDICT_INFUTURE']
NAV_FAILURE                              = runner_settings['NAV_FAILURE']
SUCCESS                                  = runner_settings['SUCCESS']
LIGHT_BLUE_COLOR                         = runner_settings['COLOR']


class BaseRunnerDisplay():
    """Base class for procedure runners."""

    def setup(self, x_bounds, y_bounds,
              arena,
              noise_sigma_x,
              noise_sigma_y,
              spaceship):
        """Create base class version of fcn to initialize procedure runner."""
        pass

    def clear_visualization(self, t):
        """Create base class version of timestep start for procedure runner."""
        pass

    def update_asteroid_state(self, idx, x, y):
        pass

    def asteroid_estimated_at_loc(self, idx, x, y, is_match=False):
        pass

    def asteroid_estimates_compared(self, num_matched, num_total):
        pass

    def spaceship_at_loc(self, h, laser_len=None):
        pass

    def laser_target_heading(self, rad, laser_len):
        pass

    def estimation_done(self, retcode, t):
        pass

    def update_visualization(self, t):
        pass

    def teardown(self, set_pos=False):
        pass

    def laser_destruct(self):
        pass


def add_observation_noise(asteroid_locations, noise_sigma_x=0.0,
                          noise_sigma_y=0.0):
    """Add noise to asteroid observations."""
    noisy_asteroid_locations = deepcopy(asteroid_locations)
    for idx, val in asteroid_locations.items():
        x, y = val
        err_x = random.normalvariate(mu=0.0, sigma=noise_sigma_x)
        err_y = random.normalvariate(mu=0.0, sigma=noise_sigma_y)
        x += err_x
        y += err_y
        noisy_asteroid_locations[idx] = (x, y)
    return noisy_asteroid_locations


class MalformedEstimate(Exception):
    """Raise this type of exception if an estimate is not a three-tuple."""
    pass

def validate_estimate(tpl):
    """Ensure that estimate of asteroid location is valid."""
    try:
        idx, (x, y) = tpl
        return (int(idx), float(x), float(y))
    except (ValueError, TypeError) as e:
        raise MalformedEstimate('Estimated location should be of the form {idx: (x, y)}, a dictionary with type {int: (float, float)} or equivalent numpy types.')


def run_estimation(asteroidshower,
                   arena,
                   agent,
                   noise_sigma_x,
                   noise_sigma_y,
                   asteroid_match_range,
                   spaceship,
                   time_limit,
                   weights=None,
                   display=None,
                   case_id=None):
    """Run the estimation procedure."""

    # local debugging variables for estimates
    SHOW_SPECIFIC_IDS = {}  # leave empty to show all IDS (be sure to turn on DEBUG_DISPLAY when you want to show these IDS)
    noisy_trajectory = False
    realtime_metrics = False
    noise = True


    display.setup(asteroidshower.x_bounds, asteroidshower.y_bounds,
                  arena,
                  noise_sigma_x=noise_sigma_x,
                  noise_sigma_y=noise_sigma_y,
                  spaceship=spaceship, show_specific_ids=SHOW_SPECIFIC_IDS)


    # I---create datastructures
    asteroids_estimate_future = {}  # estimated locations
    asteroids_prior_ts_estimates = defaultdict(lambda: None)
    all_data = {}
    all_idx = set()
    outside_field_l_r_b = {}
    first_ts_matched = defaultdict(int)
    num_consec_matches = defaultdict(int)
    num_matches = defaultdict(int)
    max_num_consec_matches = defaultdict(int)
    count = Counter()
    asteroid_lifespan = Counter()
    all_observations = []
    asteroid_distance = defaultdict(int)
    asteroid_cuml_distance = defaultdict(int)
    fast_match_threshold = 20
    realtime_accuracy = 0
    realtime_max_consec_match = 0
    realtime_1stmatch_time = 0
    x_bounds = asteroidshower.x_bounds
    y_bounds = asteroidshower.y_bounds
    score_data = {'acc': realtime_accuracy,
                  'speed': realtime_1stmatch_time,
                  'consistent': realtime_max_consec_match,
                  }

    # II---store all asteroid current and future locations
    seen = set()
    num_time_steps_to_predict_infuture = 1
    for t in range(0, time_limit + num_time_steps_to_predict_infuture):
        locations, asteroids_outside_field, _ = asteroidshower.asteroid_locations(t, seen)
        seen.update(asteroids_outside_field)
        all_data[t] = locations
        outside_field_l_r_b[t] = asteroids_outside_field
        asteroid_lifespan.update(locations.keys())

    for k, _ in asteroid_lifespan.items():
        asteroid_lifespan[k] -= 1

    asteroid_lifespan_less1 = asteroid_lifespan

    # III---Iterate through each timestep
    for timesteps in range(time_limit):
        # input(f't:{timesteps}')
        t = timesteps

        display.clear_visualization(t, score_data=score_data, case_id=case_id)
        future_true = defaultdict(lambda: None)
        # 1-- Get asteroids true locations at required timesteps
        asteroids_true_current_ts = all_data[t]
        asteroids_true_ts_in_future = all_data[t + num_time_steps_to_predict_infuture]

        # 2a-- check if student implemented required function
        if t < 2 and asteroids_estimate_future == {-1: (5.5, 5.5)}:
            print("Ensure you have implemented the spaceship class\'s predict_from_observations function!")
            ret = (FAILURE_UNIMPLEMENTED, t)
            display.teardown(FAILURE_UNIMPLEMENTED, set_pos=x_bounds)
            return ret

        # 3-- Get true data for timestep(s) ahead
        for idx, val in asteroids_true_ts_in_future.items():
            x, y = val
            future_true[idx] = (x, y)

        # 4-- Setup true asteroid positions for current timestep, and estimated positions from prior timestep in display
        for idx, val in asteroids_true_current_ts.items():
            all_idx.add(idx)
            x, y = val
            display.update_asteroid_state(idx, x, y)
            # display a ring around the asteroid location to visualize the min dist for estimate being correct
            if DISPLAY_ASTEROID_MATCH_RANGE:
                display.update_asteroid_estimate_range(idx, x, y, size=asteroid_match_range)
            if asteroids_prior_ts_estimates[idx]:
                x_est, y_est, is_match = asteroids_prior_ts_estimates[idx]
                display.asteroid_estimated_at_loc(idx, x_est, y_est, is_match)

        # 5--Calculate noisy positions
        if not noise:
            noise_sigma_x, noise_sigma_y = 0, 0
        noisy_data = add_observation_noise(asteroids_true_current_ts, noise_sigma_x, noise_sigma_y)

        # 6-- Display trajectory??
        if noisy_trajectory:
            for idx, val in noisy_data.items():
                xx, yy = val
                observation_idx = -idx - (t * 100)
                all_observations.append((observation_idx, xx, yy))
                display.update_asteroid_state(idx, x, y)

            for observation_idx, x, y in all_observations:
                display.update_asteroid_state(observation_idx, x, y, noisy_data=True)

        # -- Show true asteroid positions and estimates for current timestep in display
        display.update_visualization(t)
        #7 Get estimated positions for future timestep from student
        asteroids_estimate_future = spaceship.predict_from_observations(deepcopy(noisy_data))

        # 7b-- Validation check. check return len match input
        if len(asteroids_estimate_future) != len(noisy_data):
            print("Ensure you provide estimates for each input value!")
            ret = (FAILURE_UNIMPLEMENTED, t)
            display.teardown(FAILURE_UNIMPLEMENTED, set_pos=x_bounds, y_bounds=y_bounds)
            return ret

        # 8-- Compare estimated current position. Store matching results. Display estimates
        for tpl in asteroids_estimate_future.items():
            # Check format of result
            idx, x_est, y_est = validate_estimate(tpl)

            if idx not in future_true:
                display.clear_asteroid(idx)
                continue
            # compute the distance between the asteroid's estimated location and its true location
            x_true, y_true = future_true[idx]
            dist = distance_formula((x_est, y_est), (x_true, y_true))

            asteroid_distance[idx] = dist
            # Estimate matches, if distance between true and estimate location are less than asteroid_match_range
            is_match = (dist <= asteroid_match_range)
            count[idx] += 1

            if is_match:
                # Check for 1st time that specific asteroid was matched
                # This gives a 10 timestep grace period after which the points periodically drop each timestep from 10 to 0
                if not first_ts_matched[idx]:
                    #
                    val = count[idx]
                    first_ts_matched[idx] = val
                num_consec_matches[idx] += 1
                num_matches[idx] += 1
            else:
                num_consec_matches[idx] = 0

            max_num_consec_matches[idx] = max(max_num_consec_matches[idx], num_consec_matches[idx])
            asteroids_prior_ts_estimates[idx] = (x_est, y_est, is_match)
        for idx, _ in asteroids_true_current_ts.items():
            asteroid_cuml_distance[idx] += asteroid_distance[idx]

        if realtime_metrics:
            if num_matches:
                pct_accuracy = stats.mean(
                num_matches[idx] / asteroid_lifespan_less1[idx] for idx in all_idx if
                asteroid_lifespan_less1[idx])
            else:
                pct_accuracy = 0
            if max_num_consec_matches:
                num_matches_consc = stats.mean(
                max_num_consec_matches[idx] / asteroid_lifespan_less1[idx] for idx in all_idx if
                asteroid_lifespan_less1[idx])
            else:
                num_matches_consc = 0
            if first_ts_matched:
                time_1st_match = stats.mean(first_ts_matched.values())
            else:
                time_1st_match = 0
            score_data['acc'] = round(pct_accuracy, 2)
            score_data['speed'] = round(time_1st_match, 2)
            score_data['consistent'] = round(num_matches_consc, 2)



    # IV Calculate scoring results for testcase
    scores_quick_match = []
    max_fast_score = 10
    for idx, val in first_ts_matched.items():
        val = fast_match_threshold - first_ts_matched[idx]
        result = clamp(val, 0, max_fast_score)
        scores_quick_match.append(result)
    if scores_quick_match:
        score_fast = stats.mean(scores_quick_match) / max_fast_score
    else:
        score_fast = 0
    if max_num_consec_matches:
        score_consistency = stats.mean(
        max_num_consec_matches[idx] / asteroid_lifespan_less1[idx] for idx in all_idx if
        asteroid_lifespan_less1[idx])  # How consistently the matches occur
    else:
        score_consistency = 0
    if num_matches:
        score_accuracy = stats.mean(num_matches[idx] / asteroid_lifespan_less1[idx] for idx in all_idx if
                                asteroid_lifespan_less1[idx])  # How long the matches occur for   20/30
    else:
        score_accuracy = 0

    score_fast *= 10
    score_consistency *= 40
    score_accuracy *= 50
    overall_score = sum((score_fast, score_consistency, score_accuracy))

    if overall_score >= 20:
        ret = (SUCCESS, t)
        message = SUCCESS
    else:
        ret = (FAILURE_LOW_SCORE, time_limit)
        message = FAILURE_LOW_SCORE
    display.estimation_done(*ret)
    display.teardown(message, set_pos=x_bounds, y_bounds=y_bounds)
    return ret


def run_jumps(asteroidshower,
                    arena,
                    agent,
                    noise_sigma_x,
                    noise_sigma_y,
                    asteroid_match_range,
                    spaceship,
                    time_limit,
                    weights=None,
                    display=None,
                    case_id=None):
    """Run Jump procedure."""


    ret = (NAV_FAILURE, time_limit)
    x_bounds = asteroidshower.x_bounds
    y_bounds = asteroidshower.y_bounds
    spaceship_jump_distance = agent.jump_distance
    SHOW_SPECIFIC_IDS = {}  # leave empty to show all IDS (be sure to turn on DEBUG_DISPLAY when you want to show these IDS)


    display.setup(x_bounds, y_bounds,
              arena,
              noise_sigma_x=noise_sigma_x,
              noise_sigma_y=noise_sigma_y,
              spaceship=spaceship,
              goal_line=True,
              show_specific_ids=SHOW_SPECIFIC_IDS)

    message = FAILURE_TIME_STEP_LIMIT_EXCEEDED
    x_start, y_start = agent.get_agent_position(t=0)
    xypos_start = (x_start, y_start)
    display.agent_at_loc(x_start, y_start, size=spaceship_jump_distance)
    display.update_visualization(0)

    agent_data = {'ridden_asteroid': None,
                  'jump_distance': agent.jump_distance,
                  }
    noise = True
    noisy_trajectory = False
    all_data = {}
    outside_field_l_r_b = {}
    seen = set()
    double_jump = False
    display_args = {}
    end_display = False
    previous_ridden_idx = None


    for t in range(time_limit+NUM_TIMESTEPS_TO_PREDICT_INFUTURE):
        locations, asteroids_outside_field, outside_sides_bottom= asteroidshower.asteroid_locations(t, seen)
        seen.update(asteroids_outside_field)
        all_data[t] = locations
        outside_field_l_r_b[t] = outside_sides_bottom

    # Display parameters at initialization
    display_args['t'] = 0
    display_args['curr_agent_pos'] = agent.get_agent_position()
    display_args['asteroids_true_current_ts'] = all_data[0]
    display_args['spaceship_jump_distance'] = spaceship_jump_distance
    display_args['case_id'] = case_id
    update_all_visualization(display, display_args, simulation_start=True)

    for t in range(time_limit):
        full_loop = False

        # input(f't:{t}')
        # step 1. Get all necessary stored Data for asteroid and agent (for current and future timesteps)
        asteroids_true_current_ts = all_data[t]
        asteroids_true_ts_in_future = all_data[t + NUM_TIMESTEPS_TO_PREDICT_INFUTURE]
        asteroids_outside_field_l_r_b = outside_field_l_r_b[t + NUM_TIMESTEPS_TO_PREDICT_INFUTURE]
        ridden_asteroid = agent_data["ridden_asteroid"]
        agent_location_at_start_of_timestep = agent.get_agent_position()


        # step 2. Add Noise to current timestep data
        asteroids_noisy_coordinates = add_observation_noise(asteroids_true_current_ts, noise_sigma_x, noise_sigma_y)

        # step 3. Pass Noisy data to student and obtain student decision to jump/not jump at the targeted timestep
        selected_idx, asteroids_estimated_coordinates = spaceship.jump(deepcopy(asteroids_noisy_coordinates), deepcopy(agent_data))

        # step 4. Validate student return
        if not isinstance(selected_idx, int):
            if selected_idx is not None:
                print("Return must be of types: boolean or Int. Ensure you have implemented the necessary function.")
                message = FAILURE_UNIMPLEMENTED
                ret = (message, t)

                break

        # step 5. Check correctness of decision to jump/not jump
        # If jumping in targeted future timestep, the intended asteroid should be in jump distance, and both the intended asteroid
        # and spaceship should be in the field. If not jumping. The spaceship should not go out of bounds in the targeted timestep.
        # data validation. Make sure student doesnt try to jump to sth


        # Jumping
        if selected_idx is not None:
            if selected_idx in asteroids_true_ts_in_future:
                asteroid_xypos_fut = asteroids_true_ts_in_future[selected_idx]
                if ridden_asteroid:
                    if selected_idx == ridden_asteroid:
                        print("You are jumping to the same asteroid that you are already on, please return None instead.")
                    if ridden_asteroid in asteroids_true_ts_in_future:
                        double_jump = False
                        spaceship_xypos_fut = asteroids_true_ts_in_future[ridden_asteroid]
                    else:
                        # Failed. spaceship outside bounds. Too late to jump
                        message = FAILURE_AGENT_OUT_OF_BOUNDS
                        ret = (message, t)

                        end_display = True
                        break
                else:
                    double_jump = True
                    # if not riding asteroid, must be at start location
                    spaceship_xypos_fut = xypos_start

                valid_jump = asteroid_in_jump_range(spaceship_xypos_fut, asteroid_xypos_fut, spaceship_jump_distance, double_jump)
                if valid_jump:
                    if ridden_asteroid:
                        previous_ridden_idx = ridden_asteroid
                    agent_data['ridden_asteroid'] = selected_idx
                    agent.set_agent_position(*asteroid_xypos_fut)
                else:
                    message = FAILURE_ASTEROID_OUT_OF_JUMP_RANGE
                    ret = (message, t)

                    end_display = True
                    break

            # trying to jump onto sth outside field
            else:
                # Failed. targeted asteroid out of field
                message = FAILURE_AGENT_OUT_OF_BOUNDS
                ret = (message, t)

                end_display = True
                break

        # Not jumping
        else:
            # if currently on an asteroid,
            if ridden_asteroid:
                # which stays in bound...
                if ridden_asteroid in asteroids_true_ts_in_future:
                    # keep riding updating the spacesip position!
                    asteroid_xypos_fut = asteroids_true_ts_in_future[ridden_asteroid]
                    agent.set_agent_position(*asteroid_xypos_fut)

                else:
                    # Success as agent reached from top
                    if ridden_asteroid not in asteroids_outside_field_l_r_b:
                        message = SUCCESS
                        ret = (SUCCESS, t)

                    # failure as agent went out of bounds
                    else:
                        message = FAILURE_AGENT_OUT_OF_BOUNDS
                        ret = (message, t)
                    end_display = True
                    break
            else:
                # Keep waiting
                pass

        # step 6. Check for Success
        agent_y = agent.get_agent_position(t)[1]
        ret, message = spaceship_reached_goal(agent_y, y_bounds[1], t, ret)
        if message == SUCCESS:
            end_display = True
            break

        # step 7a. Gather required display inputs
        display_args['t'] = t
        display_args['case_id'] = case_id
        display_args['jump_to_idx'] = selected_idx
        display_args['noisy_trajectory'] = noisy_trajectory
        display_args['asteroid_match_range'] = asteroid_match_range
        display_args['ridden_asteroid'] = ridden_asteroid
        display_args['jump_departure_location'] = agent_location_at_start_of_timestep
        display_args['jump_destination_location'] = asteroids_true_ts_in_future.get(selected_idx, asteroids_true_current_ts.get(selected_idx))
        display_args['spaceship_jump_distance'] = spaceship_jump_distance
        display_args['estimates'] = asteroids_estimated_coordinates
        display_args['asteroids_true_ts_in_future'] = asteroids_true_ts_in_future
        display_args['asteroids_true_current_ts'] = asteroids_true_current_ts
        display_args['previously_ridden_idx'] = previous_ridden_idx
        display_args['message'] = message
        display_args['x_bounds'] = x_bounds
        display_args['y_bounds'] = y_bounds
        display_args['end_display'] = end_display
        display_args['curr_agent_pos'] = agent.get_agent_position()


        # step 7b. Display visualization results
        update_all_visualization(display, display_args)
        full_loop = True

    if not full_loop:
        display_args['t'] = t
        display_args['case_id'] = case_id
        display_args['jump_to_idx'] = selected_idx
        display_args['noisy_trajectory'] = noisy_trajectory
        display_args['asteroid_match_range'] = asteroid_match_range
        display_args['ridden_asteroid'] = ridden_asteroid
        display_args['jump_departure_location'] = agent_location_at_start_of_timestep
        display_args['jump_destination_location'] = asteroids_true_ts_in_future.get(selected_idx, asteroids_true_current_ts.get(selected_idx))
        display_args['spaceship_jump_distance'] = spaceship_jump_distance
        display_args['estimates'] = asteroids_estimated_coordinates
        display_args['asteroids_true_ts_in_future'] = asteroids_true_ts_in_future
        display_args['asteroids_true_current_ts'] = asteroids_true_current_ts
        display_args['previously_ridden_idx'] = previous_ridden_idx
        display_args['message'] = message
        display_args['x_bounds'] = x_bounds
        display_args['y_bounds'] = y_bounds
        display_args['end_display'] = end_display
        display_args['curr_agent_pos'] = agent.get_agent_position()

        # step 7b. Display visualization results
        update_all_visualization(display, display_args)


    display.update_visualization(t)
    display.teardown(message, set_pos=x_bounds, y_bounds=y_bounds)
    return ret


def spaceship_reached_goal(agent_y, goal, t, ret=None):
    message = FAILURE_TIME_STEP_LIMIT_EXCEEDED
    ret = (message, t)
    if agent_y >= goal - .01:
        ret = (SUCCESS, t)
        message = SUCCESS
    return ret, message


def update_all_visualization(display, display_args, simulation_start=False):
    if simulation_start:
        t = display_args['t']

        agent_xy = display_args['curr_agent_pos']
        asteroids_true_current_ts = display_args['asteroids_true_current_ts']
        spaceship_jump_distance = display_args['spaceship_jump_distance']
        display.agent_at_loc(*agent_xy, size=spaceship_jump_distance)
        for idx, true_xy in asteroids_true_current_ts.items():
            display.update_asteroid_state(idx, *true_xy)
        display.update_visualization(t)

    else:
        t = display_args['t']
        case_id = display_args['case_id']
        selected_idx = display_args['jump_to_idx']
        noisy_trajectory = display_args['noisy_trajectory']
        asteroid_match_range = display_args['asteroid_match_range']
        ridden_asteroid = display_args['ridden_asteroid']
        spaceship_jump_distance = display_args['spaceship_jump_distance']
        estimates = display_args['estimates']
        asteroids_true_ts_in_future = display_args['asteroids_true_ts_in_future']
        asteroids_true_current_ts = display_args['asteroids_true_current_ts']
        previous_ridden_idx = display_args['previously_ridden_idx']
        message = display_args['message']
        x_bounds = display_args['x_bounds']
        y_bounds = display_args['y_bounds']
        end_display = display_args['end_display']
        agent_xy = display_args['curr_agent_pos']
        jump_departure_location = display_args['jump_departure_location']
        jump_destination_location = display_args['jump_destination_location']



        display.clear_visualization(t, case_id=case_id)
        display.agent_at_loc(*agent_xy, size=spaceship_jump_distance)

        # Stop asteroids moving out of bounds from piling up in display edges by removing them
        for idx in asteroids_true_current_ts:
            if idx not in asteroids_true_ts_in_future:
                display.clear_asteroid(idx)

        # If jumping from a ridden asteroid revert prior asteroid to original color
        if selected_idx and ridden_asteroid:
                if previous_ridden_idx:
                    display.update_asteroid_color(previous_ridden_idx, change_color=True)


        for idx, true_xy in asteroids_true_ts_in_future.items():
            if idx == selected_idx:
                display.clear_agent()
                display.update_asteroid_state(idx, *true_xy, new_color=LIGHT_BLUE_COLOR, change_color=True)
                display.agent_at_loc(*true_xy, size=spaceship_jump_distance)
            else:
                display.update_asteroid_state(idx, *true_xy)

            if estimates and idx in estimates:
                estimated_xy = estimates[idx]
                x_est, y_est = estimated_xy
                dist = distance_formula((x_est, y_est), true_xy)
                is_match = (dist < asteroid_match_range)
                display.asteroid_estimated_at_loc(idx, x_est, y_est, is_match, curr_ridden=ridden_asteroid)
                if DISPLAY_ASTEROID_MATCH_RANGE:
                    display.update_asteroid_estimate_range(idx, *true_xy, size=asteroid_match_range)
                if DISPLAY_ESTIMATED_SPACESHIP_MATCH_RANGE:
                    if ridden_asteroid == idx:
                        display.estimated_agent_at_loc(idx, x_est, y_est, size=spaceship_jump_distance, is_match=is_match)
                    if idx == selected_idx:
                        display.estimated_agent_at_loc(idx, x_est, y_est, size=spaceship_jump_distance, is_match=is_match)
            if ridden_asteroid == idx:
                display.clear_agent()
                display.agent_at_loc(*true_xy, size=spaceship_jump_distance)

        if selected_idx is not None and previous_ridden_idx != selected_idx:
            if DISPLAY_ORIGIN_TO_DESTINATION_LINE:
                display.draw_attempted_jump_line(jump_departure_location, jump_destination_location,
                                                 t, point_distance=asteroid_match_range)

        display.update_visualization(t)
        # If an asteroid is moving out of bounds(i.e. not in fut), ensure the visualization deletes the object.


        if end_display:
            display.teardown(message, set_pos=x_bounds, y_bounds=y_bounds)

def asteroid_in_jump_range(agent_fut_pos, asteroid_fut_pos, spaceship_jump_distance, double_jump):
    jump = False
    if double_jump:
        spaceship_jump_distance *= 2
        # spaceship_jump_distance *= 1
    distance_agent_to_asteroid = distance_formula(agent_fut_pos, asteroid_fut_pos)
    if distance_agent_to_asteroid <= spaceship_jump_distance:
        jump = True
    return jump
