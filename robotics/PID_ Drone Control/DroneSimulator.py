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


from DualRotor import DualRotor
from DroneListener import DroneListener
from drone_pid import pid_thrust, pid_roll
import traceback
                                            

class DroneSimulator:
    def __init__(self):
        self.listeners: list(DroneListener) = []
        self.bad_listeners = []
    
    def initialize(self, test_thrust = False, test_roll=False, target_elevation: list = [], 
                target_x: list = [], simulation_length=300, target_hover_time=10, supply_params=False, 
                drone_mass=500, drone_rpm_error=0, target_elev_error = 0.0067, 
                target_x_error = 0.00, wind_factor = 0.0, wind_speed_x = 0.0, test_integral = False, 
                plan:dict = {}, ignore_collision = True, 
                DEBUG=False, VISUALIZE=False):
        
        self.test_thrust        = test_thrust
        self.test_roll          = test_roll
        #self.target_xy          = target_xy
        self.target_elevation   = target_elevation
        self.target_x           = target_x
        self.drone_mass         = drone_mass
        self.simulation_length  = simulation_length
        self.target_hover_time  = target_hover_time
        self.supply_params      = supply_params
        self.target_elev_error  = target_elev_error
        self.target_x_error     = target_x_error
        self.drone_rpm_error    = drone_rpm_error
        self.wind_factor        = wind_factor
        self.wind_speed_x       = wind_speed_x
        self.test_integral      = test_integral
        self.plan               = plan
        self.ignore_collision   = ignore_collision
        self.DEBUG              = DEBUG
        self.VISUALIZE          = VISUALIZE
        
        self.drone_rpm_left     = []
        self.drone_rpm_right    = []
        
        self.drone         = None
        self.drone_max_vel = 0
        
        self.drone_x = []
        self.drone_y = []
        
        # Actual time steps taken by drone to reach targets
        # Stored in the sequence of targets defined in self.plan
        self.actual_time = []
        
        self.hover_error    = 0
        self.path_error     = 0
        self.velocity_error = 0
        self.total_osc      = 0 # total oscillations
        
        self.extra_load     = 0
        self.motor_drift    = 0
        
        self.score = 0
        
        self.runtime_error  = False
        
    def add_listener(self, listener: DroneListener):
        self.listeners.append(listener)
            
        
    def run(self, thrust_params: dict = {}, 
                  roll_params: dict = {},
                  DEBUG = None, 
                  VISUALIZE = None,
                  FINAL_RUN = False):
        
        #print("Starting Simulator.run() ", mproc.current_process().name)
        #time.sleep(5)
        
        if DEBUG == None:
            DEBUG = self.DEBUG
        if VISUALIZE == None:
            VISUALIZE = self.VISUALIZE
          
        self.drone              = DualRotor()
        drone                   = self.drone
        drone.mass              = self.drone_mass
        drone.RPM_ERROR         = self.drone_rpm_error
        drone.WIND_FACTOR_STD   = self.wind_factor
        drone.WIND_SPEED_X      = self.wind_speed_x
        
        if ('max_propeller_speed' in self.plan):
            drone.MAX_PROPELLER_SPEED = self.plan['max_propeller_speed']
            
        if(FINAL_RUN and 'extra_load' in self.plan):
            self.extra_load = float(self.plan['extra_load'])
            drone.mass += self.extra_load
            drone.weight = drone.mass * drone.g
            self.extra_load_carry_time = int(self.plan['extra_load_carry_time']) * 10
        
        if('motor_drift' in self.plan):
            drone.MOTOR_DRIFT = float(self.plan['motor_drift'])
            
        if VISUALIZE:
            for l in self.listeners:
                if l not in self.bad_listeners:
                    try:
                        l.initialize(self.target_elevation, self.target_x, self.simulation_length, self.plan['path'])
                    except:
                        self.bad_listeners.append(l)
                    
        osc                 = 0 # osciallations per segment
        errors              = []
        self.hover_error    = 0
        self.path_error     = 0
        self.velocity_error = 0
        self.drone_max_vel  = 0
        self.total_osc      = 0 # total oscillations
        prev_sign_y         = None
        prev_sign_x         = None
        thrust              = 0
        roll                = 0
        data_thrust         = dict() #{'max_rpm': drone.MAX_PROPELLER_SPEED*2}
        data_roll           = dict() #{'max_roll_angle': drone.MAX_ROLL_ANGLE, 'max_rpm': drone.MAX_PROPELLER_SPEED*2}
        
        self.drone_x        = []
        self.drone_y        = []
        
        self.runtime_error  = False

        for t in range(self.simulation_length):
            
            if FINAL_RUN and self.extra_load > 0:
                if t >= self.extra_load_carry_time:
                    drone.mass -= self.extra_load
                    drone.weight = drone.mass * drone.g
                    self.extra_load= 0

            if self.test_thrust:
                try:
                    if self.test_integral:
                        # test integral only uses student supplied gain values

                        thrust, data_thrust = pid_thrust(self.target_elevation[t], drone.get_elevation(),
                                                         thrust_params['tau_p'], thrust_params['tau_d'],
                                                         thrust_params['tau_i'], data_thrust)
                        # roll, data_roll     = part_3_pid_roll(target_x, drone.get_x_coord(), roll_params['tau_p'], roll_params['tau_d'], roll_params['tau_i'], data_roll)

                        # data_thrust['prev_thrust'] = thrust
                        # if thrust_params.get('tau_i', 0) != 0 and \
                        if drone.is_max_rpm_reached():  # and \
                            # thrust > 0:

                            data_thrust['max_rpm_reached'] = True
                            # thrust = 0
                        else:
                            data_thrust['max_rpm_reached'] = False

                    elif self.supply_params:
                        thrust, data_thrust = pid_thrust(self.target_elevation[t], drone.get_elevation(),
                                                         thrust_params['tau_p'], thrust_params['tau_d'],
                                                         thrust_params['tau_i'], data_thrust)
                        
                except Exception as e:
                    print("There is an error in the pid_thrust function.", e)
                    self.runtime_error = True
                    break
                
                
            if self.test_roll:
                '''
                if self.test_integral:
                    # Roll is not tested for integral at this time
                    #pass
                    
                    data_roll['prev_roll'] = roll
                    #if roll_params.get('tau_i', 0) != 0:
                    if (drone.roll_angle == drone.MAX_ROLL_ANGLE and roll > 0) or \
                       (drone.roll_angle == -drone.MAX_ROLL_ANGLE  and roll < 0):
                        
                        data_roll['control_saturated'] = True
                        roll = 0
                    else:
                        data_roll['control_saturated'] = False
                '''
                
                #elif self.supply_params:
                try:
                    roll, data_roll = pid_roll(self.target_x[t], drone.get_x_coord(), roll_params['tau_p'],
                                               roll_params['tau_d'], roll_params['tau_i'], data_roll)
                except Exception as e:
                    print("There is an error in the pid_roll function.", e)
                    self.runtime_error = True
                    break
                    
                data_roll['prev_roll'] = roll
                #if roll_params.get('tau_i', 0) != 0:
                #if (drone.roll_angle == drone.MAX_ROLL_ANGLE and roll > 0) or \
                #   (drone.roll_angle == -drone.MAX_ROLL_ANGLE  and roll < 0):
                    
                #     data_roll['control_saturated'] = True
                    #roll = 0
                #else:
                #    data_roll['control_saturated'] = False

                 
            drone.update_rotor_speed(thrust, roll)
            drone.move()
            
            if drone.velocity > self.drone_max_vel:
                self.drone_max_vel = drone.velocity
            
            # If collision check is on and collision is detected, 
            # then end simulation
            if not self.ignore_collision and self.detect_collision(drone):
                if VISUALIZE:
                    for l in self.listeners:
                        l.end_simulation()
                        
                return self.hover_error, float(self.plan.get('max_velocity', 0)), self.drone_max_vel, self.plan['max_oscillations'], self.total_osc

            
            if drone.get_x_coord() >= self.target_x[t] and \
                    drone.get_elevation() >= self.target_elevation[t]:
                
                self.actual_time.append(t)
                
            error = 0
            y_error = 0
            x_error = 0
            
            if self.test_thrust:
                y_error = (self.target_elevation[t] - drone.get_elevation())
                
                #if t >= 1:
                #y_change = (drone.get_elevation() - self.drone_y[t-1])
            
                if abs(y_error) >= 0.001: #1e-2:
                    sign = y_error/abs(y_error)
                    
                    if prev_sign_y != None and sign != prev_sign_y:
                        osc += 1
                    
                    prev_sign_y = sign
                    
            if self.test_roll:
                x_error = (self.target_x[t] - drone.get_x_coord())
                
                #if t >= 1:
                #x_change = (drone.get_x_coord() - self.drone_x[t-1])
                
                if abs(x_error) >= 0.001: #1e-2: # != 0:
                    sign = x_error/abs(x_error)
                    
                    if prev_sign_x != None and sign != prev_sign_x:
                        osc += 1
                
                    prev_sign_x = sign
            
            error = abs(y_error) + abs(x_error)
            errors.append(error)
            
            self.drone_x.append(drone.get_x_coord())
            self.drone_y.append(drone.get_elevation())
            
            if self.target_x[t] != self.target_x[t-1] or \
                    self.target_elevation[t] != self.target_elevation[t-1]:
                
                self.total_osc += osc
                osc        = 0
                prev_sign_y = None
                prev_sign_x = None
                                
            
            if DEBUG==2:
                print("Step " + str(t) + \
                      ": y, x, roll, propeller_speed, thrust, weight = ", \
                      drone.get_elevation(), drone.get_x_coord(), drone.roll_angle, drone.propeller_speed, \
                      drone.thrust, drone.weight)
                
            
            if VISUALIZE:
                for l in self.listeners:
                    if l not in self.bad_listeners:
                        try:
                            l.update(drone.get_x_coord(), drone.get_elevation(), drone.roll_angle, thrust, roll, \
                                     drone.propeller_speed[1], drone.propeller_speed[2],{'pid_thrust':thrust_params, 'pid_roll':roll_params})
                        except:
                            self.bad_listeners.append(l)
            
            self.drone_rpm_left.append(drone.propeller_speed[1])
            self.drone_rpm_right.append(drone.propeller_speed[2])
            
        if VISUALIZE:
            for l in self.listeners:
                if l not in self.bad_listeners:
                    l.end_simulation()
                
        self.total_osc += osc # last sgement's oscillations are not added inside the loop, so add them here
        
        if DEBUG:
            print("thrust params: ", thrust_params)
            print("roll_params: ", roll_params)
        
        self.calculate_errors(errors, self.total_osc, DEBUG)
        
        return self.hover_error, float(self.plan.get('max_velocity', 0)), self.drone_max_vel, self.plan['max_oscillations'], self.total_osc
                
            
    def calculate_errors(self, errors, total_osc, DEBUG=False):
        
        self.hover_error = self.get_hovering_error(errors, DEBUG)
        self.path_error = self.get_path_error(DEBUG)
        
        if DEBUG:
            print("path_error, hover_error, max_vel, total_osc = ", self.path_error, self.hover_error, self.drone_max_vel, total_osc)
        

    def get_hovering_error(self, errors, DEBUG=False):
        hover_error = 0
        segment_len = 0
        for i in range(len(self.plan['target_time'])):
            hover_time = self.plan['hover_time'][i] * 10
            segment_len += (self.plan['target_time'][i] * 10) + hover_time
            segment_errors = errors[segment_len-hover_time:segment_len]
            num_segment_errors = len(segment_errors)
            if num_segment_errors:
                hover_error += sum(abs(se) for se in segment_errors) / num_segment_errors
            if DEBUG:
                print("segment_len, hover_time", segment_len, hover_time)
        
        return hover_error
    
    def get_path_error(self, DEBUG=False):
        path_error = 0
        prev_x = 0
        prev_y = 0
        
        segment_start = 0
        for i in range(len(self.plan['target_time'])):
            segment_len = (self.plan['target_time'][i] + self.plan['hover_time'][i]) * 10
            x = self.plan['path'][i][0]
            y = self.plan['path'][i][1] #['y'][i]
            
            if y != prev_y: #y changed
                # measure error in x
                segment_drone_x = self.drone_x [ 
                                        segment_start  :  \
                                        segment_start  +  segment_len 
                                  ]
                segment_error = [x - x2 for x2 in segment_drone_x]
                path_error += sum(segment_error)
            
            if x != prev_x: #x changed
                # measure error in y
                segment_drone_y = self.drone_y [ 
                                        segment_start  :  \
                                        segment_start  +  segment_len 
                                  ]
                segment_error = [y - y2 for y2 in segment_drone_y]
                path_error += sum(segment_error)
            
            prev_x = x
            prev_y = y
            segment_start += segment_len
        
        return path_error

    def detect_collision(self):
        path = self.plan['path']
        for i in range(1, len(path)): #path['x'])):
            #if self.path['x'][i-1] == self.path['x'][i]:
            if path[i-1][0] == path[i][0]:
                x2 = path[i-1][0] + 2
            else:
                x2 = path[i][0]
                
            if path[i-1][1] == path[i][1]:
                y2 = path[i-1][1] - 2
            else:
                y2 = path[i][1]
                
            if path[i-1][0] <= self.drone.x <= x2 and \
                self.path[i-1][1] >= self.drone.y >= y2:
                
                return True
        
        return False
        

                        
    