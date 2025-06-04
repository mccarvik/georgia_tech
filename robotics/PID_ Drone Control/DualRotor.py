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

import math
import random

class DualRotor(object):
    g = 9.8 # gravity = 9.8 m/s2
        
    def __init__(self):
        self.y = 0 # 0 is ground
        self.x = 0 # 0 is starting point
        
        self.propeller_speed  = {1:0, 2:0} # rpm for each propeller. 1=left, 2=right
        
        self.thrust     = {1:0, 2:0} # thrust at each propeller
        self.mass       = 2 #30 #kg
        self.weight     = self.mass * self.g
        self.velocity   = 0
        self.y_velocity = 0
        self.x_velocity = 0
        self.length     = 0.25 # meters. Side to side length of drone
        self.roll_angle = 0 # The drone's normal resting position is 0 roll angle.
                            # For now it is pointing right (along x-axis). This 
                            # is significant when calculating roll angles.
        
        # Simple displacement added to the 
        # x-axis movement of the drone as a result
        # of wind speed.
        
        # TODO: Replace with motion model according to wind
        # speed
        self.WIND_FACTOR_STD = 0.0 # std dev for wind
        self.WIND_SPEED_X = 0.0 # a constant wind speed in the horizontal direction                         
        
        self.RPM_ERROR = 0 #2 # propeller RPM drifts by this amount
        
        self.MAX_PROPELLER_SPEED = 10000
        self.MAX_RPM_CHANGE_PER_STEP = 10 #500
        #self.MAX_THRUST = 2 * self.weight
        self.RPM_TO_THRUST_RATIO = 2 #1.1
        self.RPM_TO_ROLL_RATIO = 1 #0.05
        self.MAX_ROLL_ANGLE = 0.33 * math.pi
        self.MAX_HEIGHT = 1000 #meters
        self.MOTOR_DRIFT = 0 #diff betwee right and left motor speeds
        
        self.sensor_noise_x = 0
        self.sensor_noise_y = 0
    
    def increase_propeller_speed(self, id, rpm=10):
        assert rpm >= 0, "rpm must be 0 or positive"
        rpm = min(self.MAX_RPM_CHANGE_PER_STEP, rpm)
        
        self.propeller_speed[id] = min(self.propeller_speed[id] + rpm, self.MAX_PROPELLER_SPEED)
        
        # update thrust based on a simplified formula
        # let's say the rpm to thrust force ratio is 1:1.2
        self.thrust[id] = self.propeller_speed[id] / self.RPM_TO_THRUST_RATIO
        
    def decrease_propeller_speed(self, id, rpm=10):
        assert rpm >= 0, "rpm must be 0 or positive"
        rpm = min(self.MAX_RPM_CHANGE_PER_STEP, rpm)
        self.propeller_speed[id] = max(self.propeller_speed[id] - rpm, 0)
        
        # update thrust based on a simplified formula
        # let's say the rpm to thrust force ratio is 1:1.2
        self.thrust[id] = self.propeller_speed[id] / self.RPM_TO_THRUST_RATIO
        
    def add_thrust(self, thrust=5, roll=0):
        self.thrust[id] = min(self.thrust[id] + thrust, self.MAX_THRUST)
    
    def reduce_thrust(self, id, thrust=10):
        self.thrust[id] = max(self.thrust[id] - thrust, 0)
    
    def set_thrust(self, id, net_force):
        self.thrust[id] = net_force + self.g/2
        self.propeller_speed[id] = self.thrust[id] * self.RPM_TO_THRUST_RATIO
    
    def get_elevation(self):
        return self.y + self.sensor_noise_y
    
    def get_x_coord(self):
        return self.x + self.sensor_noise_x
    
    def get_coordinates(self):
        return (self.x + self.sensor_noise_x, self.y + self.sensor_noise_y)
    
    def is_max_rpm_reached(self):
        return (self.propeller_speed[1] + self.propeller_speed[2]) == self.MAX_PROPELLER_SPEED*2
    
    def is_max_roll_reached(self):
        return abs(self.roll_angle) == self.MAX_ROLL_ANGLE
    
    def wind_speed(self):
        if self.WIND_FACTOR_STD != 0:
            return random.gauss(0, self.WIND_FACTOR_STD)
        else:
            return self.WIND_SPEED_X
    
    def calc_one_side_move(self, id, dt):
        # calculate net force: total thrust by propeller - weight of drone
        F = self.thrust[id] - self.weight/2 # Assume each propeller supports 1/2 weight. 
                                            # Net force, can be negative (downward) or positive (upward)
            
        # calculate acceleration
        a = F / self.mass
        
        # calculate movement
        new_y = self.y + (1/2) * a * dt**2 + self.y_velocity * dt - 0.1 * self.y_velocity
        
        new_y = max(new_y, 0)
        
        
        # Calculate x and y components of thrust.
        # Zero roll angle is pointing right (along x-axis)
        # so tilt right will be negative, and since we are 
        # calculating angles from drone being in upright position
        # (i.e. 90 degrees), we will add the angle to 90 degrees
        # for tilt right.
        #
        # Similarly tilt left will have positive roll
        #
        x_disp = new_y * math.cos((math.pi/2) + self.roll_angle)
        
        new_y = new_y * math.sin((math.pi/2) + self.roll_angle)
        new_y = max(new_y, 0) # sin might be negative
        
                
        # create some invisible bound for x, if needed
        # self.x = max(min(self.x, 2), -2)
        
        return new_y, x_disp, a
    
        
    def move(self, dt=1/10):
        
        # Time is 1/10 sec by default
        
        propeller_speeds_same = (self.propeller_speed[1] == self.propeller_speed[2])
        
        if propeller_speeds_same:
            # calculate net force: total thrust by propeller - weight of drone
            F = self.thrust[1] + self.thrust[2] - self.weight #net force, can be negative (downward) or positive (upward)
            
            # calculate acceleration
            a = F / self.mass
            
            # calculate movement
            new_y = self.y + (1/2) * a * dt**2 + self.y_velocity * dt - 0.1 * self.y_velocity
            
            new_y = max(new_y, 0)
            
            #self.y_velocity = (new_y - self.y) / dt
            self.y_velocity = self.y_velocity + a * dt
            
            #print("Propeller speed, y_velocity, a, F, mass: ", self.propeller_speed, self.y_velocity, a, F, self.mass)
            
            new_x = self.x
            
            #add wind factor but only if drone has lift off
            if new_y > 0:
                new_x += self.wind_speed()
            
            self.velocity = math.sqrt((new_y - self.y)**2 + (new_x - self.x)**2) / dt
            
            
            self.y = new_y
            self.x = new_x
            
        else:  # propeller speeds are not the same
            
            # first calculate new_y for each side
            new_y1, x_disp_1, a1 = self.calc_one_side_move(1, dt) # left side
            new_y2, x_disp_2, a2 = self.calc_one_side_move(2, dt) # right side
            
            new_y = 0.5 * (new_y1 + new_y2) # center point of the drone, approx as average of each side's y
            
            #self.velocity = (new_y - self.y) / dt
            self.y_velocity = self.y_velocity + (a1 + a2) * dt
            
            
            new_x1 = self.x + x_disp_1 - self.length # left side
            new_x2 = self.x + x_disp_2 + self.length # right side
            
                
            # find the slope (roll)
            # m = (new_y2 - new_y1)/(new_x2 - new_x1)
            
            # find roll angle
            self.roll_angle = math.atan2(new_y2 - new_y1, new_x2 - new_x1)
            # For now 0 is pointing right (along x-axis)
            # so tilt right will be negative
            
            new_x = self.x + x_disp_1 + x_disp_2 # + self.x_velocity * dt - 0.1 * self.x_velocity
            
            #add wind factor but only if drone has lift off
            if new_y > 0:
                new_x += self.wind_speed()
            
            self.velocity = math.sqrt((new_y - self.y)**2 + (new_x - self.x)**2) / dt
            #self.x_velocity = (new_x - self.x) / dt
            
            self.y = new_y
            self.x = new_x
            
        
            
    
    def update_rotor_speed(self, thrust_chg, roll_chg):
        '''
        thrust - Change in thrust
        roll - change in roll
        '''
        
        # Clip thrust change to max thrust change allowed per step
        if thrust_chg != 0:
            sign = thrust_chg/abs(thrust_chg)
            thrust_chg = min(self.MAX_RPM_CHANGE_PER_STEP / self.RPM_TO_THRUST_RATIO, abs(thrust_chg))
            thrust_chg *= sign
        else:
            thrust_chg += self.RPM_ERROR/self.RPM_TO_THRUST_RATIO
        
        # Clip roll change to max roll change allowed
        if roll_chg != 0:
            roll_sign = roll_chg / abs(roll_chg)
            roll_chg = min(self.MAX_ROLL_ANGLE, abs(roll_chg))
            roll_chg *= roll_sign
        
        #print("adjusted thrust = ", thrust)
        #print("adjusted roll = ", roll)
        
        #self.propeller_speed[1] = min(max(0, self.propeller_speed[1] + thrust_chg*self.RPM_TO_THRUST_RATIO - roll_chg*self.RPM_TO_ROLL_RATIO + self.RPM_ERROR), self.MAX_PROPELLER_SPEED)
        #self.propeller_speed[2] = min(max(0, self.propeller_speed[2] + thrust_chg*self.RPM_TO_THRUST_RATIO + roll_chg*self.RPM_TO_ROLL_RATIO + self.RPM_ERROR), self.MAX_PROPELLER_SPEED)
        self.propeller_speed[1] = min(self.propeller_speed[1] + thrust_chg*self.RPM_TO_THRUST_RATIO - roll_chg*self.RPM_TO_ROLL_RATIO + self.RPM_ERROR, self.MAX_PROPELLER_SPEED)
        self.propeller_speed[2] = min(self.propeller_speed[2] + thrust_chg*self.RPM_TO_THRUST_RATIO + roll_chg*self.RPM_TO_ROLL_RATIO + self.RPM_ERROR, self.MAX_PROPELLER_SPEED)
        self.propeller_speed[1] -= self.propeller_speed[1]*self.MOTOR_DRIFT
        self.propeller_speed[2] += self.propeller_speed[2]*self.MOTOR_DRIFT
        
        self.thrust[1] = self.propeller_speed[1]/self.RPM_TO_THRUST_RATIO
        self.thrust[2] = self.propeller_speed[2]/self.RPM_TO_THRUST_RATIO