
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
import sys
import time
import runner
from runner import BaseRunnerDisplay
from settings import visualizer_settings
try:
    import turtle
except ModuleNotFoundError:
    turtle = None

# get/import required settings
LIGHT_COLOR_PALETTE = visualizer_settings['LIGHT_COLOR_PALETTE']
DARK_COLOR_PALETTE  = visualizer_settings['DARK_COLOR_PALETTE']
COLOR_PALETTE       = visualizer_settings['COLOR_PALETTE']
DEBUG_DISPLAY       = visualizer_settings['DEBUG_DISPLAY']
DARK_MODE           = visualizer_settings['DARK_MODE']
sleep_constant      = visualizer_settings['SLEEP_CONSTANT']


class TurtleRunnerDisplay(runner.BaseRunnerDisplay):
    """Handle GUI display of the asteroids project simulation."""

    def __init__(self, width, height, x_bounds, y_bounds):
        """Initialize the turtle display."""
        self.width = width
        self.height = height
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.asteroid_turtles = {}
        self.asteroid_observation_turtles = {}
        self.estimated_asteroid_turtles = {}
        self.asteroid_estimation_range_turtles = {}

        self.asteroid_estimation_range_turtles2 = {}
        self.jump_contrails = {}
        self.estimated_agent = {}

        self.time_turtle = turtle.Turtle()
        self.case_id = turtle.Turtle()
        self.acc_metric = turtle.Turtle()
        self.speed_metric = turtle.Turtle()
        self.consistency_metric = turtle.Turtle()
        self.toggle_label = turtle.Turtle()
        self.asteroid_size = 0.3
        self.agent_size = 1
        self.rocket_size = 0.02
        self.agent_created = False
        self.estimated_agent_created = False
        self.spaceship_objects = []
        self.boundary_variable = 0.2 # this is a temp variable for adjusting the display boundary
        self.my_asteroid_size_ = 0.02  # this is a temp variable to standardize some settings across asteroids
        self.toggle_x_offset = 0.12
        self.toggle_y_offset = 0.12
        self.toggle_circle1_x_offset = 0.06
        self.toggle_circle2_x_offset = 0.06
        self.canvas_data = [self.time_turtle, self.acc_metric, self.speed_metric, self.consistency_metric, self.case_id, self.toggle_label]


    def setup(self, x_bounds, y_bounds,
              arena,
              noise_sigma_x,
              noise_sigma_y,
              spaceship,
              goal_line=False,
              show_specific_ids=None):
        """Initialize necessary turtles and their properties."""

        self.show_specific_ids = show_specific_ids
        BOTTOM_TEXT_DISPLAY_Y_OFFSET = 0.08
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        xmin, xmax = x_bounds
        ymin, ymax = y_bounds
        dx = xmax - xmin
        dy = ymax - ymin
        turtle.setup(width=self.width, height=self.height, startx=10, starty=50)
        self.canvas_lowerl_x = xmin - (self.boundary_variable)
        self.canvas_lowerl_y = ymin - (self.boundary_variable)
        self.canvas_upperr_x = xmax + (self.boundary_variable)
        self.canvas_upperr_y = ymax + (self.boundary_variable)
        turtle.setworldcoordinates(self.canvas_lowerl_x,
                                   self.canvas_lowerl_y,
                                   self.canvas_upperr_x,
                                   self.canvas_upperr_y)
        self.pixel_constant = self.pixel_unit_conversion()
        turtle.tracer(0, 1)
        if goal_line:
            self._draw_inbounds(arena, color_goal=True)
        else:
            self._draw_inbounds(arena)
        self.time_turtle.penup()
        self.time_turtle.setposition(self.x_bounds[0] + self.boundary_variable/2, self.y_bounds[0] - BOTTOM_TEXT_DISPLAY_Y_OFFSET)
        self.time_turtle.hideturtle()
        self.acc_metric.penup()
        self.acc_metric.setposition(self.x_bounds[0] + self.boundary_variable*3, self.y_bounds[0] - BOTTOM_TEXT_DISPLAY_Y_OFFSET)
        self.acc_metric.hideturtle()
        self.speed_metric.penup()
        self.speed_metric.setposition(self.x_bounds[0] + self.boundary_variable*3 +0.05, self.y_bounds[0] - BOTTOM_TEXT_DISPLAY_Y_OFFSET*2)
        self.speed_metric.hideturtle()
        self.consistency_metric.penup()
        self.consistency_metric.setposition(self.x_bounds[0] + self.boundary_variable*3 + 0.12, self.y_bounds[0] - BOTTOM_TEXT_DISPLAY_Y_OFFSET*3)
        self.consistency_metric.hideturtle()
        self.case_id.penup()
        self.case_id.setposition(self.x_bounds[1] - self.boundary_variable / 2,
                                 self.y_bounds[1] + BOTTOM_TEXT_DISPLAY_Y_OFFSET)
        self.case_id.hideturtle()
        self.toggle_label.penup()
        self.toggle_label.setposition(self.x_bounds[0] + self.boundary_variable * 2,
                                 self.y_bounds[1] + BOTTOM_TEXT_DISPLAY_Y_OFFSET)
        self.toggle_label.hideturtle()

        turtle.hideturtle()
        turtle.bgcolor(COLOR_PALETTE['background_color'])

    def _draw_inbounds(self, arena, color_goal=False):
        """Draw the bounds of the simulation world."""
        self.border = turtle.Turtle()
        self.border.hideturtle()
        line_thickness = 5
        self.border.pensize(line_thickness)
        # COLOR
        self.border.pencolor(COLOR_PALETTE['border'])
        self.border.penup()
        self.border.setposition(arena.x_bounds[0], arena.y_bounds[0])
        self.border.pendown()
        self.border.setposition(arena.x_bounds[1], arena.y_bounds[0])
        if DEBUG_DISPLAY:
            self.border._write('({}, {})'.format(arena.x_bounds[1], arena.y_bounds[0]), 'left', 'arial')
        self.border.setposition(arena.x_bounds[1], arena.y_bounds[1])
        if DEBUG_DISPLAY:
            self.border._write('({}, {})'.format(arena.x_bounds[1], arena.y_bounds[1]), 'left', 'arial')
        if color_goal:
            self.border.pencolor(COLOR_PALETTE['border_goal'])
        self.border.setposition(arena.x_bounds[0], arena.y_bounds[1])
        self.border.pencolor(COLOR_PALETTE['border'])
        if DEBUG_DISPLAY:
            self.border._write('({}, {})'.format(arena.x_bounds[0], arena.y_bounds[1]), 'right', 'arial')
        self.border.setposition(arena.x_bounds[0], arena.y_bounds[0])
        if DEBUG_DISPLAY:
            self.border._write('({}, {})'.format(arena.x_bounds[0], arena.y_bounds[0]), 'right', 'arial')

        self.button = turtle.Turtle()
        self.button.penup()

        center_toggle_frame = (self.x_bounds[0]) + self.toggle_y_offset,\
            (self.y_bounds[1] + self.toggle_y_offset)
        self.button.setpos(*center_toggle_frame)
        self.button.pendown()
        self.button.shape('square')
        self.button.shapesize(2, 4)
        self.button.fillcolor('grey')

        turtle.width(100)
        # Create rectangle  ------------
        self.rect_small = turtle.Turtle()
        self.rect_small.penup()
        # self.rect_small.setpos(-.85, 0.863)
        position = (self.x_bounds[0] + self.toggle_y_offset), (self.y_bounds[1] + self.toggle_y_offset)
        self.rect_small.setpos(*position)
        self.rect_small.pendown()
        self.rect_small.shape('square')
        self.rect_small.shapesize(1, 2.2)
        self.rect_small.fillcolor(COLOR_PALETTE['rect_small'])
        self.rect_small.color(COLOR_PALETTE['rect_small'])

        # draw agent start boundary marks on x-axis
        min_xtick = arena.agent_xstart_min
        max_xtick = arena.agent_xstart_max
        self.x_ticks_min = turtle.Turtle()
        self.x_ticks_min.penup()
        self.x_ticks_min.setposition(min_xtick, arena.y_bounds[0])
        self.x_ticks_min.shape('square')
        self.x_ticks_min.shapesize(1, 0.25)
        self.x_ticks_max = turtle.Turtle()
        self.x_ticks_max.penup()
        self.x_ticks_max.setposition(max_xtick, arena.y_bounds[0])
        self.x_ticks_max.shape('square')
        self.x_ticks_max.shapesize(1, 0.25)

        # circle1----------------
        self.toggle_circle1 = turtle.Turtle()
        self.toggle_circle1.penup()
        self.toggle_circle1.shape('circle')
        # self.toggle_circle1.setpos(-0.92, 0.86)
        position = (self.x_bounds[0] + self.toggle_x_offset - self.toggle_circle1_x_offset), (self.y_bounds[1] + self.toggle_y_offset)
        self.toggle_circle1.setpos(*position)
        self.toggle_circle1.circle(80)
        self.toggle_circle1.color(COLOR_PALETTE['toggle_circle1'])
        self.toggle_circle1.fillcolor(COLOR_PALETTE['toggle_fill_circle1'])

        # circle2-------------
        self.toggle_circle2 = turtle.Turtle()
        self.toggle_circle2.penup()
        self.toggle_circle2.shape('circle')
        # self.toggle_circle2.setpos(-0.79, 0.86)
        position = (self.x_bounds[0] + self.toggle_x_offset + self.toggle_circle2_x_offset), (
                    self.y_bounds[1] + self.toggle_y_offset)
        self.toggle_circle2.setpos(*position)
        self.toggle_circle2.circle(80)
        self.toggle_circle2.color(COLOR_PALETTE['toggle_circle2'])
        self.toggle_circle2.fillcolor(COLOR_PALETTE['toggle_fill_circle2'])

        def button_toggle(x, y):
            switch_color_palette()
        self.toggle_circle1.onclick(button_toggle)
        self.toggle_circle2.onclick(button_toggle)
        self.rect_small.onclick(button_toggle)
        self.button.onclick(button_toggle)
        self.rect_small.fillcolor(COLOR_PALETTE['rect_small'])


        def switch_color_palette():
            # DARK_MODE = False
            # DARK_MODE = True
            global COLOR_PALETTE
            if COLOR_PALETTE == LIGHT_COLOR_PALETTE:
                COLOR_PALETTE = DARK_COLOR_PALETTE

            else:
                COLOR_PALETTE = LIGHT_COLOR_PALETTE
            self.toggle_circle1.color(COLOR_PALETTE['toggle_circle1'])
            self.toggle_circle1.fillcolor(COLOR_PALETTE['toggle_fill_circle1'])
            self.rect_small.fillcolor(COLOR_PALETTE['rect_small'])
            self.rect_small.color(COLOR_PALETTE['rect_small'])
            self.toggle_circle2.color(COLOR_PALETTE['toggle_circle2'])
            self.toggle_circle2.fillcolor(COLOR_PALETTE['toggle_fill_circle2'])
            self.time_turtle.color(COLOR_PALETTE['label_text'])
            self.case_id.color(COLOR_PALETTE['label_text'])
            self.border.pencolor(COLOR_PALETTE['border'])
            turtle.bgcolor(COLOR_PALETTE['background_color'])

            self.border.penup()
            self.border.setposition(arena.x_bounds[0], arena.y_bounds[0])
            self.border.pendown()
            self.border.setposition(arena.x_bounds[1], arena.y_bounds[0])
            if DEBUG_DISPLAY:
                self.border._write('({}, {})'.format(arena.x_bounds[1], arena.y_bounds[0]), 'left', 'arial')
            self.border.setposition(arena.x_bounds[1], arena.y_bounds[1])
            if DEBUG_DISPLAY:
                self.border._write('({}, {})'.format(arena.x_bounds[1], arena.y_bounds[1]), 'left', 'arial')
            self.border.setposition(arena.x_bounds[0], arena.y_bounds[1])
            if DEBUG_DISPLAY:
                self.border._write('({}, {})'.format(arena.x_bounds[0], arena.y_bounds[1]), 'right', 'arial')
            self.border.setposition(arena.x_bounds[0], arena.y_bounds[0])
            if DEBUG_DISPLAY:
                self.border._write('({}, {})'.format(arena.x_bounds[0], arena.y_bounds[0]), 'right', 'arial')

            for _, met_trtl in self.asteroid_turtles.items():
                met_trtl.color(COLOR_PALETTE['asteroid'])

    def clear_visualization(self, t, score_data=None, add_tail=False, case_id=None):
        """Set up turtles for the beginning of current timestep t."""
        acc, speed, consistent = 0, 0, 0
        if score_data:
            acc = score_data['acc']
            speed = score_data['speed']
            consistent = score_data['consistent']
        for val in self.canvas_data:
            val.clear()
            val.hideturtle()
            val.color(COLOR_PALETTE['label_text'])

        self.time_turtle._write(f"Time: {t}", 'center', 'arial')
        self.case_id._write(f"Test Case: {case_id}", 'center', 'arial')
        self.toggle_label._write(f"Light/Dark", 'center', 'arial')
        if acc:
            self.acc_metric._write(f"Acc(Avg): {acc}", 'center', 'arial')
        if speed:
            self.speed_metric._write(f"1st match(Avg): {speed}", 'center', 'arial')
        if consistent:
            self.consistency_metric._write(f"Max cons match(Avg): {consistent}", 'center', 'arial')

        if add_tail:
            pass
        else:
            for idx, trtl in list(self.asteroid_turtles.items()):
                trtl.clear()
            for idx, trtl in list(self.estimated_asteroid_turtles.items()):
                trtl.clear()
            for idx in self.asteroid_estimation_range_turtles:
                self.asteroid_estimation_range_turtles[idx].clear()
            for spaceship in self.spaceship_objects:
                spaceship.clear()
            self.spaceship_objects.clear()
            if self.estimated_agent:
                self.estimated_agent.clear()

        self.clear_agent()


    def clear_agent(self):

        if self.agent_created:
            self.agent.clear()
            self.agent.hideturtle()
            self.agent_spaceship.clear()
            self.agent_spaceship.hideturtle()


    def clear_asteroid(self, idx):
        if idx in self.asteroid_turtles:
            self.asteroid_turtles[idx].clear()
            self.asteroid_turtles[idx].hideturtle()

        if idx in self.estimated_asteroid_turtles:
            self.estimated_asteroid_turtles[idx].clear()
            self.estimated_asteroid_turtles[idx].hideturtle()

        if idx in self.asteroid_estimation_range_turtles:
            self.asteroid_estimation_range_turtles[idx].clear()
            self.asteroid_estimation_range_turtles[idx].hideturtle()
        self.delete_asteroid(idx)

    def delete_asteroid(self, i):
        # Only display estimates in bound

        if i in self.asteroid_turtles:
            del self.asteroid_turtles[i]
        if i in self.estimated_asteroid_turtles:
            del self.estimated_asteroid_turtles[i]
        if i in self.asteroid_estimation_range_turtles:
            del self.asteroid_estimation_range_turtles[i]
        return


    def pixel_unit_conversion(self):
        '''This function converts the units needed into the proportional number of pixels required when
        using certain turtle methods e.g. turtle.shapesize(). The constant '20' is the default representing
        1 unit. See the documentation for more information.'''

        pixel_to_canvas_unit_ratio = 20
        canvas_width = self.canvas_upperr_x - self.canvas_lowerl_x
        constant = self.width/ (canvas_width * pixel_to_canvas_unit_ratio)
        return constant

    def create_spaceship(self, x, y):
        '''Code modified from '''
        rocket = turtle.Turtle()
        x_delta = 0.02
        y_delta = 0.03
        rocket.penup()
        rocket.color(COLOR_PALETTE['agent'])
        rocket.goto(x-x_delta, y - self.rocket_size-y_delta)
        rocket.pendown()

        rocket.forward(x_delta*2)
        rocket.setheading(65)
        rocket.circle(0.15, 65, steps=50)
        rocket.setheading(230)
        rocket.circle(0.15, 65, steps=50)
        rocket.penup()
        rocket.goto(x-x_delta, y - self.rocket_size-y_delta-0.007)
        rocket.setheading(0)
        rocket.pendown()
        rocket.forward(x_delta*2)
        rocket.setheading(-65)
        rocket.forward(x_delta*2)
        rocket.setheading(180)
        rocket.forward(x_delta * 3.8)
        rocket.setheading(65)
        rocket.forward(x_delta*2)
        rocket.penup()
        # rocket.hideturtle()
        return rocket

    def agent_at_loc(self, x, y, size=None):
        """Display Spaceship at provided location."""
        # Draw the agent

        if size == None:
            size = self.agent_size

        if not self.agent_created:
            # rocket = self.create_spaceship(x, y)
            # self.agent_spaceship = rocket
            size = self.pixel_constant * size
            trtl = turtle.Turtle()
            trtl.shape("circle")
            trtl.shapesize(size*2, size*2)
            trtl.color(COLOR_PALETTE['agent'])
            # trtl.color('purple')
            trtl.fillcolor('')
            trtl.penup()
            self.agent = trtl
            self.agent_created = True
        self.agent.setposition(x, y)

        self.agent_spaceship = self.create_spaceship(x, y)
        self.agent_spaceship.setposition(x, y)
        self.agent.showturtle()
        self.agent_spaceship.hideturtle()
        self.spaceship_objects.append(self.agent_spaceship)

    def estimated_agent_at_loc(self, idx, x, y, size=None, is_match=False):
        """Display asteroid at provided location."""
        # Draw the agent
        agent_color = COLOR_PALETTE['agent_estimated_correct'] if is_match else COLOR_PALETTE['agent_estimated_incorrect']
        if size == None:
            size = self.agent_size

        if not self.estimated_agent_created:
            size = self.pixel_constant*size
            trtl = turtle.Turtle()
            trtl.shape("circle")
            trtl.shapesize(size*2, size*2)
            trtl.color(agent_color)
            trtl.fillcolor('')
            trtl.penup()
            self.estimated_agent = trtl
            self.estimated_agent_created = True
        self.estimated_agent.color(agent_color)
        self.estimated_agent.fillcolor('')
        self.estimated_agent.setposition(x, y)


    def update_asteroid_estimate_range(self, idx, x, y, size=None):
        '''

        :param idx:
        :param x:
        :param y:
        :param size:
        :return:
        '''

        if idx not in self.asteroid_estimation_range_turtles:
            size = self.pixel_constant*size
            trtl = turtle.Turtle()
            # trtl.circle(size)
            trtl.shape("circle")
            # size += self.asteroid_size
            trtl.shapesize(2*size, 2*size)
            trtl.color(COLOR_PALETTE['asteroid_range'])
            trtl.fillcolor('')
            trtl.penup()
            self.asteroid_estimation_range_turtles[idx] = trtl
        self.asteroid_estimation_range_turtles[idx].setposition(x, y)
        self.asteroid_estimation_range_turtles[idx].showturtle()

    def update_asteroid_color(self, i, new_color=COLOR_PALETTE['asteroid'], change_color= False):
        '''

        :param i:
        :param new_color:
        :param change_color:
        :return:
        '''
        if change_color:
            if i in self.asteroid_turtles:
                self.asteroid_turtles[i].clear()
                self.asteroid_turtles[i].color(new_color)
                if DEBUG_DISPLAY:
                    if not self.show_specific_ids or i in self.show_specific_ids:
                        self.asteroid_turtles[i]._write(str(i), 'center', 'arial')

    def update_asteroid_state(self, i, x, y, new_color=COLOR_PALETTE['asteroid'], change_color= False, noisy_data=False):
        """Display asteroid at provided location."""
        # create turtle object if not created
        size = self.asteroid_size
        # size = self.pixel_constant * size

        if noisy_data:

            # if i not in self.asteroid_observation_turtles:
            trtl = turtle.Turtle()
            trtl.shape("circle")
            trtl.shapesize(size, size)
            trtl.color("blue")
            trtl.penup()
            self.asteroid_observation_turtles[i] = trtl
            trtl.setposition(x, y)
            self.asteroid_observation_turtles[i].showturtle()

        else:
            if i not in self.asteroid_turtles:
                trtl = turtle.Turtle()
                trtl.shape("circle")
                trtl.shapesize(size, size)
                trtl.color(new_color)
                trtl.penup()
                self.asteroid_turtles[i] = trtl
            if change_color:
                self.asteroid_turtles[i].color(new_color)
            self.asteroid_turtles[i].setposition(x, y)
            # if DEBUG_DISPLAY:
            #     self.asteroid_turtles[i]._write(str(i), 'center', 'arial')
            self.asteroid_turtles[i].showturtle()
            # self.asteroid_turtles[i].hideturtle()


    def asteroid_estimated_at_loc(self, i, x, y, is_match=False, curr_ridden=None):
        """Display asteroid estimate at provided location.

        asteroid will be colored green if it is close enough to the
        asteroid's true position to be counted as correct, and will be colored
        red otherwise.
        """

        # Only display estimates in bound
        if x < self.x_bounds[0] or x > self.x_bounds[1]:
            # self.delete_asteroid(i)

            return

        if y < self.y_bounds[0] or y > self.y_bounds[1]:
            # self.delete_asteroid(i)

            return


        if i not in self.estimated_asteroid_turtles:
            trtl = turtle.Turtle()
            trtl.shape("circle")
            if i is curr_ridden:
                # if the spaceship is currently riding this very asteroid
                trtl.color(COLOR_PALETTE['estimated_asteroid_loc_correct'] if is_match else COLOR_PALETTE['agent_estimated_incorrect'])
            else:
                trtl.color(COLOR_PALETTE['estimated_asteroid_loc_correct'] if is_match else COLOR_PALETTE['estimated_asteroid_loc_incorrect'])

            trtl.shapesize(self.asteroid_size, self.asteroid_size)
            trtl.penup()
            self.estimated_asteroid_turtles[i] = trtl


        # COLOR
        if i is curr_ridden:
            # if the spaceship is currently riding this very asteroid color asteroid right color and agent wrong color respectively
            self.estimated_asteroid_turtles[i].color(COLOR_PALETTE['estimated_asteroid_loc_correct'] if is_match else COLOR_PALETTE['agent_estimated_incorrect'])
        else:
            self.estimated_asteroid_turtles[i].color(COLOR_PALETTE['estimated_asteroid_loc_correct'] if is_match else COLOR_PALETTE['estimated_asteroid_loc_incorrect'])
        self.estimated_asteroid_turtles[i].setposition(x, y)
        self.estimated_asteroid_turtles[i].showturtle()


    def draw_attempted_jump_line(self, jump_departure_location, jump_destination_location, t, point_distance=None):
        """Draw a line between the currently ridden asteroid and the newly selected idx asteroid to visualize attempted jump"""
        if point_distance:
            spacing_between_dots = point_distance
        else:
            spacing_between_dots = .3/10

        jump_animation_speed = .05  #lower number is faster
        trtl = turtle.Turtle()
        trtl.penup()
        trtl.color('blue')
        trtl.setposition(jump_departure_location)
        x1, y1 = jump_departure_location
        x2, y2 = jump_destination_location
        to_angle = math.degrees(math.atan2(y2-y1, x2-x1))
        trtl.setheading(to_angle)
        dist_between_points = math.sqrt((y2-y1)**2+(x2-x1)**2)
        drawn_dist = 0
        spacing_between_dots = .3/10
        while drawn_dist < dist_between_points:
            trtl.dot()
            time.sleep(jump_animation_speed)
            self.update_visualization(t)
            trtl.forward(spacing_between_dots)
            drawn_dist+=spacing_between_dots
        self.jump_contrails[t] = trtl


    def update_visualization(self, t, start=False):
        """Update GUI for the end of a timestep."""
        for i in self.asteroid_turtles:
            if DEBUG_DISPLAY:
                if not self.show_specific_ids or i in self.show_specific_ids:

                    self.asteroid_turtles[i]._write(str(i), 'center', 'arial')
        turtle.update()
        time.sleep(sleep_constant)

        time_delay_to_continue_displaying_contrail = 5
        for t_created, trtl in self.jump_contrails.items():
            if t > t_created + time_delay_to_continue_displaying_contrail:
                trtl.hideturtle()
                trtl.clear()
                turtle.reset()


        if start:
            self.time_turtle.clear()
            self.time_turtle.hideturtle()
            self.acc_metric.clear()
            self.acc_metric.hideturtle()
            self.speed_metric.clear()
            self.speed_metric.hideturtle()
            self.consistency_metric.clear()
            self.consistency_metric.hideturtle()
            self.case_id.clear()
            self.case_id.hideturtle()


    def teardown(self, message='Game Over!', set_pos=False, y_bounds=None):
        """Conclude the GUI visualization of the simulation."""
        xpos = 0.2

        val1 = {1: 0.2,
               2: .7,
               4: 1.9}

        val2 = {1: 0.05,
               2: .05,
               4: 1.1}

        if set_pos:
            if len(message) <= 10:
                xpos = val1[set_pos[1]]
            else:
                xpos = val2[set_pos[1]]
        ypos = 1
        if y_bounds:
            ypos = 0.5 * y_bounds[1]
        self.exit_graphic = turtle.Turtle()
        self.exit_graphic.hideturtle()
        self.exit_graphic.setpos(xpos, ypos)
        self.exit_graphic._write(' '.join(message.split()), 'left', ('arial', 35, "normal", "bold"))
        toggle_buttons = [self.button, self.toggle_circle1, self.toggle_circle2, self.rect_small]
        for btn in toggle_buttons:
            btn.onclick(
                lambda x, y: None, 1, add=False
            )
        turtle.done()



class TextRunnerDisplay(BaseRunnerDisplay):

    def __init__(self, fout=None):
        self.fout = fout

    def _log(self, s):
        fout = self.fout or sys.stdout
        if hasattr(self, 't'):
            fout.write("[t {0:.1f}]  {1}\n".format(self.t, s))
        else:
            fout.write("{0}\n".format(s))

    def setup(self, x_bounds, y_bounds,
              arena,
              noise_sigma_x,
              noise_sigma_y,
              spaceship,
              goal_line=False,
              show_specific_ids=None):
        # self._log(f"{noise_sigma_x}  {noise_sigma_y}")
        self.t = 0

    def clear_visualization(self, t, score_data=None, add_tail=True, case_id=None):
        self.t = t


    def clear_agent(self):
        pass

    def clear_asteroid(self, idx):
        pass

    def delete_asteroid(self, idx):
        pass

    def pixel_unit_conversion(self):
        pass
    # def remove_asteroid(self, idx):
    #     pass
    def create_spaceship(self, x, y):
        pass

    def agent_at_loc(self, x, y, size=None):
        pass

    def estimated_agent_at_loc(self, idx, x, y, size=None, is_match=False):
        pass

    def update_asteroid_estimate_range(self, idx, x, y, size=None):
        pass

    def update_asteroid_color(self, i, new_color=False, change_color=False):
        pass

    def update_asteroid_state(self, i, x, y, new_color=False, change_color=False):
        pass

    def asteroid_estimated_at_loc(self, i, x, y, is_match=False, curr_ridden=None):
        pass

    def asteroid_estimates_compared(self, num_matched, num_total):
        self._log("estimates matching: {0} / {1}".format(num_matched, num_total))
        pass

    def estimation_done(self, retcode, t):
        pass
        # self._log("estimation done: {0}".format(retcode))

    def draw_attempted_jump_line(self, jump_departure_location, jump_destination_location, t, point_distance=None):
        pass

    def update_visualization(self, t, start=False):
        pass

    def teardown(self, message=None, set_pos=False, y_bounds=None):
        pass
