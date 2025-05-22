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
import collections
import io
import multiprocessing as mproc
import random
import sys
import time
import traceback
from copy import deepcopy

import runner
from visualizer import TextRunnerDisplay
from settings import test_settings
from game_objects import Arena, Asteroid, AsteroidShower, Agent
from spaceship import Spaceship
from visualizer import TurtleRunnerDisplay
import argparse
import json

def load_cases():
    cases = {}
    base_dir = 'cases'
    for case_file in sorted(os.listdir(base_dir)):
        if case_file.endswith('.json'):
            case_id = os.path.splitext(os.path.basename(case_file))[0]
            with open(os.path.join(base_dir, case_file)) as f:
                cases[case_id] = json.load(f)
    return cases

cases = load_cases()

def display_for_name(dname, params=None):
    """Set up desired display tuple."""
    x_bounds = params["arena_x_bounds"]
    y_bounds = params["arena_y_bounds"]

    canvas_xsize = x_bounds[1] * 800/2
    canvas_ysize = y_bounds[1] * 800/2
    if dname == 'turtle':
        import turtle
        return TurtleRunnerDisplay(canvas_xsize, canvas_ysize, x_bounds, y_bounds)
        # return TurtleRunnerDisplay(400, 400)
    elif dname == 'text':
        return TextRunnerDisplay()
    else:
        return runner.RunnerDisplay()

def case_params(case_num):
    """Get the parameters for the given case."""
    case_id = f'case{case_num}' if isinstance(case_num, int) else case_num
    return cases[case_id]


def run_method(method_name):
    """Convert input method into the function needed to run that method."""
    if method_name == 'estimate':
        return runner.run_estimation
    elif method_name == 'jump':
        return runner.run_jumps
    else:
        raise RuntimeError('unknown method %s' % method_name)


def run_kwargs(params):
    """Set up kwargs for running main."""
    asteroids = []
    for theasteroid in params['asteroids']:
        asteroids.append(
            Asteroid(theasteroid))
    random.seed(params['_args']['observation_noise_seed'])

    arena = Arena(params['arena_x_bounds'], params['arena_y_bounds'], agent_xstart_min_max=params['agent_x_min_max'])
    agent = Agent(x_pos=params['agent_x_min_max'], y_pos=0, jump_distance=params["agent_jump_distance"], x_bounds=params['arena_x_bounds'], y_bounds=params['arena_y_bounds'])
    x_start, y_start = agent.get_agent_position(t=0)
    spaceship = Spaceship(deepcopy(arena.bounds), (x_start, y_start))

    ret = {'asteroidshower': AsteroidShower(arena, params['_args']['observation_noise_seed'],
                                           asteroids, spaceship),
           'arena': arena,
           'agent': agent,
           'noise_sigma_x': params['noise_sigma_x'],
           'noise_sigma_y': params['noise_sigma_y'],
           'asteroid_match_range': params['asteroid_match_range'],
           'spaceship': spaceship,
           'time_limit': params['_args']['time_limit']}

    return ret

def main(method_name, case_id, display_name):
    """Run the specified case using the specified method."""

    if case_id.isdigit():
        case_id = 'case'+case_id
    try:
        params = cases[case_id]
    except Exception as e:
        print(f'fUnable to load test case: "{case_id}.json"')
        return

    retcode, t= run_method(method_name)(case_id=case_id[4:], display=display_for_name(display_name, params),
                                         **(run_kwargs(params)))

    print((retcode, t))


def parser():
    """Parse command-line arguments."""
    prsr = argparse.ArgumentParser()
    prsr.add_argument('--method',
                      help="Which method to test",
                      type=str,
                      choices=('estimate', 'jump'),
                      default='estimate')
    prsr.add_argument('--case',
                      help="test case id (one of %s) (just number is ok if test case begins with 'case') or test case file" % list(cases.keys()),
                      type=str,
                      default='1')
    prsr.add_argument('--display',
                      choices=('turtle', 'text', 'none'),
                      default='turtle')
    return prsr


try:
    # from test_one import case_params, run_method, run_kwargs
    import spaceship as spaceship
    studentExc = None
    studentTraceback = ''
except Exception as e:
    studentExc = e
    studentTraceback = traceback.format_exc()

# get/import required settings
FAILURE_EXCEPTION = test_settings['FAILURE_EXCEPTION']
ESTIMATE_TIMEOUT  = test_settings['ESTIMATE_TIMEOUT']
JUMP_TIMEOUT   = test_settings['JUMP_TIMEOUT']
FAILURE_TIMEOUT   = test_settings['FAILURE_TIMEOUT']
file_checker      = test_settings['FILE_CHECKER']

GradingTask = collections.namedtuple('GradingTask',
                                     ('case_num', 'method_name', 'weight', 'timeout'))



TASKS = (
         GradingTask(1, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(2, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(3, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(4, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(5, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(6, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(7, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(8, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(9, 'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(10,'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(11,'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(12,'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(13,'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(14,'estimate', 1, ESTIMATE_TIMEOUT),
         GradingTask(15, 'estimate', 1, ESTIMATE_TIMEOUT),

         GradingTask(16, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(17, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(18, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(19, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(20, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(21, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(22, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(23, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(24, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(25, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(26, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(27, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(28, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(29, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(30, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(31, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(32, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(33, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(34, 'jump', 1, JUMP_TIMEOUT),
         GradingTask(35, 'jump', 1, JUMP_TIMEOUT),
)
def truncate_runlog(runlog, begin_lines=10, end_lines=10):
    lines = runlog.splitlines()
    if len(lines) <= begin_lines + end_lines:
        return str(runlog)
    else:
        return '\n'.join(lines[:begin_lines] + lines[-end_lines:])


class SingleCaseGrader():

    def __init__(self):
        # Using a Manager here to create the Queue resolves timeout
        # issue on Windows.
        self.result_queue = mproc.Manager().Queue(1)

    def _reset(self):
        while not self.result_queue.empty():
            self.result_queue.get()

    def run(self, method_name, case_num, weights=None):
        try:
            display = None
            msg = ''
            fout_msg = ''
            self._reset()
            display = TextRunnerDisplay(fout=io.StringIO())
            kwargs = run_kwargs(case_params(case_num))
            retcode, t = run_method(method_name)(weights=weights, display=display, **kwargs)
        except Exception as e:
            retcode = FAILURE_EXCEPTION
            t = 1000
            msg = traceback.format_exc()
        finally:
            fout_msg = '' if not display else display.fout.getvalue()
        self.result_queue.put((retcode, t, fout_msg + '\n' + msg))

# multiprocessing_flag = False
multiprocessing_flag = True

class MultiCaseGrader():

    def __init__(self,
                 fout,
                 tasks=TASKS):
        self.fout = fout
        self.tasks = tuple(tasks)


    def run(self):



        score_estimate = 0
        score_jump = 0
        num_jump_cases = 0
        max_estimate_score = 0
        max_jump_score = 0
        self.fout.write("\n")
        self.fout.write(f"----------------------Running Tests -----------------------")
        processes = []
        scgs = []
        failed_tasks = []
        for task in self.tasks:
            scg = SingleCaseGrader()
            test_process = mproc.Process(target=scg.run,
                    args=(task.method_name, task.case_num, task.weight))
            processes.append(test_process)
            if multiprocessing_flag:
                test_process.start()

            scgs.append(scg)
            # task_names.append(tasks_)
        for scg, process, task in zip(scgs, processes, self.tasks):
            test_process = process
            runlog = ''


            try:
                if not multiprocessing_flag:
                    test_process.start()
                test_process.join(task.timeout)
            except Exception:
                retcode = FAILURE_EXCEPTION

            if test_process.is_alive():
                test_process.terminate()
                retcode = FAILURE_TIMEOUT
            else:
                if not scg.result_queue.empty():
                    retcode, t, runlog = scg.result_queue.get()
                else:
                    retcode, t, runlog = 'EXITED', -1, 'program exited '

            case_score = float(task.weight) if retcode == runner.SUCCESS else 0.
            if not case_score:
                failed_tasks.append(task.case_num)
            # self.fout.write(f"begin case {task.case_num}, method {task.method_name}")
            self.fout.write(truncate_runlog(runlog))

            self.fout.write(f"\ncase {task.case_num}, method {task.method_name}, result: {retcode}  ({case_score}/{float(task.weight)})")

            if task.method_name == 'estimate':
                score_estimate += case_score
                max_estimate_score += float(task.weight)
            else:
                score_jump += case_score
                num_jump_cases += 1
                max_jump_score += float(task.weight)

        score_jump = min(score_jump, int(0.75 * num_jump_cases))
        max_jump_score = int(0.75 * max_jump_score)
        if not max_estimate_score:
            parta_pct = 0
        else:
            parta_pct = score_estimate*100/max_estimate_score
        self.fout.write(f"\n Part A Score: {int(score_estimate)}/{int(max_estimate_score)},    {round(parta_pct, 2)}%")
        self.fout.write(f"\n")
        if not max_jump_score:
            partb_pct = 0
        else:
            partb_pct = score_jump*100/max_jump_score
        self.fout.write(f"\n Part B Score: {int(score_jump)}/{int(max_jump_score)},        {round(partb_pct, 2)}%")
        self.fout.write(f"\n")


        estimate_weight = 70
        jump_weight = 30
        if max_estimate_score == 0 and max_jump_score == 0:
            final_score = 0
        elif max_jump_score == 0:
            final_score = (score_estimate * estimate_weight/max_estimate_score)
        elif max_estimate_score == 0:
            final_score = (score_jump * jump_weight/max_jump_score)
        else:
            final_score = (score_estimate * estimate_weight/max_estimate_score) + (score_jump * jump_weight/max_jump_score)

        self.fout.write(f"\n Final Score: {round(parta_pct, 2)}% * .7(weight for Part A)   + {round(partb_pct, 2)}% * .3(weight for Part B): {int(final_score)}%")
        self.fout.write(f"\n Final Score: {round(parta_pct * 0.7, 2)}%    + {round(partb_pct * 0.3, 2)}%  : {int(final_score)}%")
        self.fout.write(f"\n")
        self.fout.write(f"----------------------Completed Running Tests -----------------------")
        self.fout.write(f'Score: {int(final_score)}')



if file_checker:
    import json
    import hashlib
    import pathlib
    print("File checking is turned on.")
    with open('file_check.json', 'r') as openfile:
        json_dict = json.load(openfile)

    modified_files = []
    for file in json_dict:
        f = str(file)
        try:
            current = pathlib.Path(file).read_text().replace(' ', '').replace('\n', '')
            file_hash = hashlib.sha256(current.encode()).hexdigest()
            if file_hash != json_dict[f]:
                modified_files.append(f)
        except:
            try:
                current = pathlib.Path(os.path.join('cases', file)).read_text().replace(' ', '').replace('\n', '')
                file_hash = hashlib.sha256(current.encode()).hexdigest()
                if file_hash != json_dict[f]:
                    modified_files.append(f)
            except:
                print(f'File ({f}) not in project folder.')

    if len(modified_files) == 0:
        print("You are running against the same runner framework as the Gradescope autograder.")
        print("Your local case files are the same as those that were posted on Canvas")
        print("at the time that your current file_check.json file was downloaded from Canvas.")
    else:
        print("Warning. The following files have been modified and the results may not be accurate:")
        print(", ".join(modified_files))


if __name__ == '__main__':
    if studentExc:
        sys.stdout.write('error importing code:\n\n')
        sys.stdout.write(studentTraceback)
        sys.stdout.write('\n')
        sys.stdout.write('score: 0\n')
    else:
        student_id = spaceship.who_am_i()
        if student_id:
            try:
                if not len(sys.argv) > 1:
                    mcg = MultiCaseGrader(sys.stdout)
                    mcg.run()
                else:
                    args = parser().parse_args()
                    if 'none' in args.display:
                        gui_runcom = sys.argv
                        gui_runcom.insert(-1, '--display turtle')
                        thecommand = '    python ' + ' '.join(gui_runcom)
                        print('No display method provided in run command; defaulting to \'text\'.')
                        print('To re-run this simulation with the GUI visualization, please run the command\n')
                        print(thecommand + '\n')
                        args.display = 'text'
                    main(method_name=args.method,
                         case_id=args.case,
                         display_name=args.display)
            except Exception as e:
                sys.stdout.write(e)
                sys.stdout.write('score: 0\n')
        else:
            sys.stdout.write("Student ID not specified.  Please fill in 'whoami' variable.\n")
            sys.stdout.write('score: 0\n')
