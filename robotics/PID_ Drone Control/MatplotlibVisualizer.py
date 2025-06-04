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


from DroneListener import DroneListener
import matplotlib.pyplot as plt

class MatplotlibVisualizer(DroneListener):
    
    
    def initialize(self, target_elevation, target_x, simulation_length, path:list() = []):
        self.target_ys = target_elevation #[target_elevation for i in range(simulation_length+1)]
        self.target_xs = target_x #[target_x for i in range(simulation_length+1)]
        self.drone_roll_angles_list  = [0]
        self.rpms_left               = [0]
        self.rpms_right              = [0]
        self.thrust_list             = [0]
        self.roll_list               = [0]
        self.xs                      = [0]
        self.ys                      = [0]
        
    def update(self, x, y, roll_angle, thrust, roll, rpm_left, rpm_right, pid_params=None):
        #plt.plot(x, y, '_', alpha=0.3)
        
        self.drone_roll_angles_list.append(roll_angle)
        self.rpms_left.append(rpm_left)
        self.rpms_right.append(rpm_right)
        self.thrust_list.append(thrust)
        self.roll_list.append(roll)
        self.xs.append(x)
        self.ys.append(y)
        
        
    def end_simulation(self, args: dict = dict()):
        plt.figure(figsize=(10, 24))
        ax1 = plt.subplot(811)
        ax1.plot(self.xs, self.ys, '_', alpha=0.3)
        ax1.set_xlim([0, 60])
        ax1.set_title('Drone flight')
        ax1.set_xlabel('Horizontal distance')
        ax1.set_ylabel('Elevation')
        
        ax2 = plt.subplot(812)
        ax2.plot(range(len(self.thrust_list)), self.thrust_list)
        ax2.set_title('Output of Thrust PID control')
        ax2.set_xlabel('Time steps')
        ax2.set_ylabel('Thrust')
        
        ax3 = plt.subplot(813)
        ax3.plot(range(len(self.roll_list)), self.roll_list)
        ax3.set_title('Ouput of Roll PID control')
        ax3.set_xlabel('Time steps')
        ax3.set_ylabel('Roll')
        
        ax4 = plt.subplot(814)
        ax4.plot(range(len(self.rpms_left)), self.rpms_left)
        ax4.set_title('Drone left propeller RPM')
        ax4.set_xlabel('Time steps')
        ax4.set_ylabel('Left RPM')
        
        ax5 = plt.subplot(815)
        ax5.plot(range(len(self.rpms_right)), self.rpms_right)
        ax5.set_title('Drone right propeller RPM')
        ax5.set_xlabel('Time steps')
        ax5.set_ylabel('Right RPM')
        
        ax6 = plt.subplot(816)
        ax6.plot(range(len(self.drone_roll_angles_list)), self.drone_roll_angles_list)
        ax6.set_title('Actual Drone roll angles')
        ax6.set_xlabel('Time steps')
        ax6.set_ylabel('Roll angle')
        
        ax7 = plt.subplot(817)
        ax7.plot(range(len(self.target_ys)), self.target_ys, label='target')
        ax7.plot(range(len(self.ys)), self.ys, label='actual')
        ax7.set_title('Target elevation vs actual')
        ax7.legend()
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Elevation')
        
        ax8 = plt.subplot(818)
        ax8.plot(range(len(self.target_xs)), self.target_xs, label='target')
        ax8.plot(range(len(self.xs)), self.xs, label='actual')
        ax8.set_title('Target horizontal position vs actual')
        ax8.legend()
        ax8.set_xlabel('Time steps')
        ax8.set_ylabel('x distance')
        
        plt.tight_layout(pad=2)
        plt.show()