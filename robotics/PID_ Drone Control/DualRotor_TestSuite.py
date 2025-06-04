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

VISUALIZE = True
VISUALIZE_TWIDDLE = False

if VISUALIZE or VISUALIZE_TWIDDLE:
    from MatplotlibVisualizer import MatplotlibVisualizer
    from TKinterVisualizer import TKinterVisualizer
    
from drone_pid import who_am_i
import unittest
import traceback
import json
import math
import multiprocessing as mproc

try: 
    from DroneSimulator import DroneSimulator
    from drone_pid import who_am_i, find_parameters_thrust, find_parameters_with_int, find_parameters_with_roll
    studentExc = None
except Exception as e:
    studentExc = traceback.format_exc()

DEBUGGING_SINGLE_PROCESS = False

DEBUG = True # This will output all timesteps for 
              # all test cases, so use with a single 
              # test case to debug, or output to a file 
              # to analyse separately.

DEBUG_TWIDDLE = False

if VISUALIZE or VISUALIZE_TWIDDLE:
    DEBUGGING_SINGLE_PROCESS = True

TWIDDLE   = True

TIME_LIMIT = 180 # secs

PART_1_a = True

Test_Cases = {'Drone Controller': PART_1_a}

class DronePIDWrapper:
    def __init__(self):
        if DEBUGGING_SINGLE_PROCESS:
            import queue
            self.thrust_params = queue.Queue(1)
            self.roll_params = queue.Queue(1)
        else:
            self.thrust_params = mproc.Manager().Queue(1)
            self.roll_params = mproc.Manager().Queue(1)
        
    def run_with_params(self, pid_func, simulator_method, tune, DEBUG_TWIDDLE, VISUALIZE_TWIDDLE):
        thrust_params, roll_params = pid_func(simulator_method, tune, DEBUG_TWIDDLE, VISUALIZE_TWIDDLE)
        self.thrust_params.put(thrust_params)
        self.roll_params.put(roll_params)

class Part_1_a_TestCase(unittest.TestCase):
    """ 
    Test Part B: Elevation and roll.
    Gain values supplied by test case
    """

    @classmethod
    def setUpClass(cls):
        """Setup test class.
        """
        cls.simulator = DroneSimulator()
        if VISUALIZE or VISUALIZE_TWIDDLE:
            #visualizer = Visualizer()
            #cls.simulator.add_listener(visualizer)
            tk = TKinterVisualizer()
            mpl = MatplotlibVisualizer()
            cls.simulator.add_listener(tk)
            cls.simulator.add_listener(mpl)
        
        
        data = None
        with open('test_cases.txt') as f:
            data = f.read()
        
        cls.test_cases = json.loads(data)
        
    def setUp(self):
        self.score = 0
        
    def spawn_process(self, find_parameters_func, simulator_run, tune, DEBUG_TWIDDLE, VISUALIZE_TWIDDLE):
        dronePID = DronePIDWrapper()
        
        test_process = mproc.Process(target=dronePID.run_with_params, args=(find_parameters_func, simulator_run, tune, DEBUG_TWIDDLE, VISUALIZE_TWIDDLE))
        thrust_params = None
        roll_params = None
        
        try:
            test_process.start()
            #print('DURING:', test_process, test_process.is_alive())
            test_process.join(TIME_LIMIT)
            #print('JOINED:', test_process, test_process.is_alive())
            
        except Exception as exp:
            error_message = exp
            print(error_message)

        if test_process.is_alive():
            test_process.terminate()
            error_message = ('Test aborted due to timeout. ' +
                             'Test was expected to finish in fewer than {} second(s).'.format(TIME_LIMIT))
            print(error_message)
        else:
            thrust_params = dronePID.thrust_params.get()
            roll_params = dronePID.roll_params.get()
        
        return thrust_params, roll_params
        
        
    def run_test(self, plan, target_y_error=0.02, target_x_error=0.02, tune="both", score_weight=1, test_integral=False, drone_rpm_error = 0, test_roll=True):
        
        #target_xy = []
        target_elev = []
        target_x    = []
        
        for i in range(len(plan['path'])): #['y'])):
            #segment = plan['path'][i]
            segment_len = (plan['target_time'][i] + plan['hover_time'][i]) *10
            #target_xy += [(segment[0], segment[1]) for j in range(segment_len)]
            target_elev += [plan['path'][i][1] for j in range(segment_len)]
            target_x    += [plan['path'][i][0] for j in range(segment_len)]

        sim_len = len(target_elev)
        
        self.simulator.initialize(
            test_thrust         = True, 
            test_roll           = test_roll,
            #target_xy           = target_xy,
            target_elevation    = target_elev,
            target_x            = target_x,
            drone_mass          = 2,
            simulation_length   = sim_len, 
            target_hover_time   = sim_len, #200, 
            supply_params       = True, 
            target_elev_error   = target_y_error, #2, #0.08, #0.02,
            target_x_error      = target_x_error, #2, #0.08, #0.02,
            plan                = plan,
            ignore_collision    = True,
            test_integral       = test_integral,
            drone_rpm_error     = drone_rpm_error,
            DEBUG               = DEBUG_TWIDDLE,
            VISUALIZE           = VISUALIZE_TWIDDLE)  
       
        thrust_params       = {'tau_p': 1, 'tau_d': 100, 'tau_i': 0} 
        roll_params         = {'tau_p': 0, 'tau_d': 0, 'tau_i': 0}
        
        runtime_error = False
        
        if TWIDDLE:
            if test_integral:
                try:
                    if DEBUGGING_SINGLE_PROCESS:
                        thrust_params, roll_params = find_parameters_with_int(self.simulator.run, tune=tune, DEBUG=DEBUG_TWIDDLE, VISUALIZE=VISUALIZE_TWIDDLE)                    
                    else:
                        thrust_params, roll_params = self.spawn_process(find_parameters_with_int, self.simulator.run, tune, DEBUG_TWIDDLE, VISUALIZE_TWIDDLE)
                            
                except Exception as e:
                    print("Error with find_parameters_with_int.", e)
                    runtime_error = True

            elif tune=="both":
                try:
                    if DEBUGGING_SINGLE_PROCESS:
                        thrust_params, roll_params = find_parameters_with_roll(self.simulator.run, tune=tune, DEBUG=DEBUG_TWIDDLE, VISUALIZE=VISUALIZE_TWIDDLE)
                    else:
                        thrust_params, roll_params = self.spawn_process(find_parameters_with_roll, self.simulator.run, tune, DEBUG_TWIDDLE, VISUALIZE_TWIDDLE)
                        
                    
                except Exception as e:
                    print("Error with find_parameters_with_roll.", e)
                    runtime_error = True
                    
            else: # tune thrust only
                try:
                    if DEBUGGING_SINGLE_PROCESS:
                        thrust_params, roll_params = find_parameters_thrust(self.simulator.run, tune=tune, DEBUG=DEBUG_TWIDDLE, VISUALIZE=VISUALIZE_TWIDDLE)
                    else:
                        thrust_params, roll_params = self.spawn_process(find_parameters_thrust, self.simulator.run, tune, DEBUG_TWIDDLE, VISUALIZE_TWIDDLE)
                        
                except Exception as e:
                    print("Error with find_parameters_thrust.", e)
                    runtime_error = True

        if runtime_error == True:
            self.score = 0
        else:
            try:
                self.simulator.run(thrust_params, roll_params, DEBUG=DEBUG, VISUALIZE=VISUALIZE, FINAL_RUN=True)
                self.score = self.get_score(target_y_error, target_x_error, plan) * score_weight
            except Exception as e:
                self.score = 0
            
    def get_score(self, target_y_error, target_x_error, plan):
        percentage_score = 0
        
        if self.simulator.runtime_error or (
                sum(self.simulator.drone_x) == 0 and 
                sum(self.simulator.drone_y) == 0
                ) or \
           math.isnan(self.simulator.hover_error):
            # Edge case if there was no movement in the drone. Without this,
            # an automated tuning algorithm may go into a wrong direction since
            # there is always going to be a positive %age score otherwise.
            percentage_score = 0
        else:
            hover_score = 0
            
            if self.simulator.hover_error == 0:
                hover_score = 1
            elif self.simulator.hover_error > 0:
                #proximity_score_pct = min(  (self.target_elev_error + self.target_x_error) / avg_error,  1.01  )
                hover_score = (target_y_error + target_x_error) / self.simulator.hover_error
                hover_score = min(hover_score, 1)
            
                
            vel_score = 0
            max_allowed_vel = float(plan.get('max_velocity', 0))
            
            if max_allowed_vel != 0 and self.simulator.drone_max_vel <= max_allowed_vel:
                vel_score = 1
            elif max_allowed_vel != 0 and self.simulator.drone_max_vel > max_allowed_vel:
                vel_score = max_allowed_vel / self.simulator.drone_max_vel
                vel_score = min(vel_score, 1)
                
            osc_score = 0
            max_allowed_osc = float(plan.get('max_oscillations', 0))
            
            if max_allowed_osc != -1 and self.simulator.total_osc <= max_allowed_osc:
                osc_score = 1
            
            elif max_allowed_osc != -1 and self.simulator.total_osc > max_allowed_osc:
                osc_score = max_allowed_osc / self.simulator.total_osc
                osc_score = min(osc_score, 1)
            
            percentage_score = (hover_score + vel_score + osc_score)/3
            percentage_score = min(max(percentage_score*100, 0), 100)
            
        return percentage_score
        
    def _test_case01(self):
        plan = self.test_cases['testcase_1']
        self.run_test(plan, target_y_error = 0.02, target_x_error=0.02, tune="thrust", score_weight=float(plan['score_weight']), test_roll=False)
        print("Test case 1 score: ", self.score)
        
    def _test_case02(self):
        plan = self.test_cases['testcase_2']
        self.run_test(plan, target_y_error = 0.02, target_x_error=0.02, tune="thrust", score_weight=float(plan['score_weight']), test_roll=False)
        print("Test case 2 score: ", self.score)
        
    def _test_case03(self):
        plan = self.test_cases['testcase_3']
        self.run_test(plan, target_y_error = 0.02, target_x_error=0.02, tune="thrust", score_weight=float(plan['score_weight']), test_roll=False)
        print("Test case 3 score: ", self.score)
        
    def _test_case04(self):
        plan = self.test_cases['testcase_4']
        drone_rpm_error = float(plan.get('rpm_error', 0))
        self.run_test(plan, target_y_error = 0.04, target_x_error=0.04, tune="thrust", test_integral=True, drone_rpm_error = drone_rpm_error, test_roll=False, score_weight=float(plan['score_weight']))
        print("Test case 4 score: ", self.score)
        
    def test_case05(self):
        plan = self.test_cases['testcase_5']
        self.run_test(plan, target_y_error = 0.035, target_x_error=0.035, tune="both", score_weight=float(plan['score_weight']))
        print("Test case 5 score: ", self.score)
        
    def _test_case06(self):
        plan = self.test_cases['testcase_6']
        self.run_test(plan, target_y_error = 0.04, target_x_error=0.04, tune="both", score_weight=float(plan['score_weight']))
        print("Test case 6 score: ", self.score)
        
    def _test_case07(self):
        plan = self.test_cases['testcase_7']
        self.run_test(plan, target_y_error = 0.04, target_x_error=0.04, tune="both", score_weight=float(plan['score_weight']))
        print("Test case 7 score: ", self.score)
        
    def _test_case08(self):
        plan = self.test_cases['testcase_8']
        self.run_test(plan, target_y_error = 0.04, target_x_error=0.04, tune="both", score_weight=float(plan['score_weight']))
        print("Test case 8 score: ", self.score)
        

class PIDTestResult(unittest.TestResult):

    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super(PIDTestResult, self).__init__(stream, verbosity, descriptions)
        self.stream = stream
        self.credit = []

    def stopTest(self, test):
        super(PIDTestResult, self).stopTest(test)
        try:
            self.credit.append(test.score)

        except AttributeError as exp:
            if self.stream != None:
                self.stream.write(str(exp))

    @property
    def avg_credit(self):
        try:
            return sum(self.credit) / len(self.credit)

        except ZeroDivisionError:
            return 0.0
        
    @property
    def total_credit(self):
        return sum(self.credit)


# This flag is used to check whether project files listed in the json have been modified.
# Modifications include (but are not limited to) print statements, changing flag values, etc.
# If you have modified the project files in some way, the results may not be accurate.
# Turn file_checker on by setting the flag to True to ensure you are running against
# the same framework as the Gradescope autograder.
file_checker = False  # set to True to turn file checking on

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
            print(f'File ({f}) not in project folder.')

    if len(modified_files) == 0:
        print("You are running against the same framework as the Gradescope autograder.")
    else:
        print("Warning. The following files have been modified and the results may not be accurate:")
        print(", ".join(modified_files))


# Only run all of the test automatically if this file was executed from the command line.
# Otherwise, let Nose/py.test do it's own thing with the test cases.
if __name__ == "__main__":
    if studentExc:
        print("Exception occurred in import: %s" % str(studentExc))
        print("score: 0")
        
    else:
        student_id = who_am_i()
        if student_id:
            overall_score = 0
            cases = []
            if PART_1_a is True: cases.append(Part_1_a_TestCase)
            
            suites = [unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(case)) for case in cases]
    
            #total_passes = 0
            #average_scores = []
            total_scores = []
            #score_dict = {'overall_score': 0}
    
            try:
                for i, suite in zip(Test_Cases.keys(), suites):
                    print("====================\nTests for {}:".format(i))
        
                    result = PIDTestResult()
                    suite.run(result)
                    #average_scores.append(result.avg_credit)
                    total_scores.append(result.total_credit)
        
                    #print("Average Weighted Score: ", result.avg_credit)
                    print("Total Weighted Score: ", result.total_credit)
        
                #overall_score = (sum(average_scores)/len(average_scores))
                overall_score = sum(total_scores)
                
                
                #myProcess = MyProcess()
                #myProcess.run_tests()
                #overall_score = myProcess.overall_score #score_dict['overall_score']
                
            except: # Exception as e:
                #print(e)
                traceback.print_exc()
                overall_score = 0
                
            if overall_score > 100:
                print()
                print("Score above 100:", overall_score, " capped to 100!")
                overall_score = 100
            print("====================\nScore: {}".format(overall_score))
        else:
            print("Student ID not specified.  Please fill in 'whoami' variable.")
            print("score: 0")
