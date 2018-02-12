import pygame
import pymunk
from pygame.locals import *
import pymunk.pygame_util
import collections
from copy import copy
from math import sqrt
import random



class Specimen(object):
    def __init__(self, parts):
        """
        Class for specimen related activities. Except from holding its parts it
        also contain methods for moving limbs and probably more.
        """
        self.parts = parts


class Simulation(object):
    def __init__(self, gravity=(0.0, 0.0), damping=0, engine_force=2000,
                 sensor_h=80, sensor_a=40, num_of_targets=5,
                 WIDTH=640, HEIGHT=480, iterations=1000, clock=0,
                 creature_spawn_loc=None, light_sources=None, use_sensor=True,
                 display=True):
        """
        Class which creates simulation, initializes world and so on.
        :param display: Specify whether initialize pygame to display simulation
        :param gravity: Gravity in pymunk
        :param sensor_h: height of triangle representing sensor field  of view
        :param sensor_a: base of triangle representing sensor field  of view
        :param iterations: Iterations for pymunk solver
        :param clock: sets value for pygame clock to maintain FPS
        :param light_sources: list of locations of so called light source.
                              Uses botom left coordinates.
        :param use_sensor: decides whether to create basic binary collistion sensor.
        """
        self.engine_force = (0, engine_force)
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.sensor_h = sensor_h
        self.sensor_a = sensor_a
        self.num_of_targets = num_of_targets
        self.display = display
        self.clock_time = clock
        self.use_sensor = use_sensor
        self.light_sources = light_sources
        self.creature_spawn_loc = creature_spawn_loc if creature_spawn_loc is not None else [WIDTH/2, HEIGHT/2]

        self.SENSOR_COLLTYPE = 1
        # Note that for different task more useful may be wildcard collision handler
        self.TARGET_COLLTYPE = 2
        self.CREATURE_COLLTYPE = 3


        # Some variable for holding sensor state (dunno how to implement sensor just yet)
        self.sensor_detected = False
        self.score = 0

        # Needed only for displaying simulation
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode([WIDTH, HEIGHT])
            pygame.display.set_caption("specimen simulation")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.space =  pymunk.Space(threaded=True)
        self.space.threads = 8
        self.space.gravity = gravity
        self.space.damping = damping
        self.space.iterations = iterations


        # Initialize simulation
        self.creature, self.trackL, self.trackR =  self.spawn_creature()
        self.trackL_center = [self.trackL.center_of_gravity.x, self.trackL.center_of_gravity.y]
        self.trackR_center = [self.trackR.center_of_gravity.x, self.trackR.center_of_gravity.y]

        self._spawn_targets([self.WIDTH, self.HEIGHT], 10)

        self.picked_up = False
        self.detected = False
        self.trackL_running = False
        self.trackR_running = False
        self.trackL_stepping = False
        self.trackR_stepping = False
        self.stepping = False
        self.timer = 0



    def _sensor_activated(self, arbiter, space, data):
        self.detected = True
        print "sensor activated!"
        return False  # pymunk should ignore this collision


    def _sensor_deactivated(self, arbiter, space, data):
        self.detected = False
        print "sensor deactivated!"


    def _pick_up_target(self, arbiter, space, data):
        self.picked_up = True
        self.score += 1

        target = arbiter.shapes[1]
        self.space.remove(target, target.body)


        # TEMP
        print self.score


        return False #  body will be removed anyway


    def _spawn_target(self, x, y, radius=10):
        body = pymunk.Body()
        body.position = [x, y]
        shape = pymunk.Circle(body, radius)
        shape.density = 1
        shape.collision_type = self.TARGET_COLLTYPE
        shape.color = [255, 70, 0]

        self.space.add(body, shape)


    def spawn_creature(self):
        body = pymunk.Body()
        body.position = self.creature_spawn_loc
        # shape = pymunk.Poly.create_box(body, (10, 40))
        trackL = pymunk.Poly(body, [[-15, 20], [-15, -20], [-5, -20], [-5, 20]])
        trackR = pymunk.Poly(body, [[15, 20], [15, -20], [5, -20], [5, 20]])
        trackL.density = 0.000002
        trackR.density = 0.000002
        trackL.color = [75, 75, 75]
        trackR.color = [75, 75, 75]
        trackL.collision_type = self.CREATURE_COLLTYPE
        trackR.collision_type = self.CREATURE_COLLTYPE

        if self.use_sensor:
            # Add sesnor
            sensor = pymunk.Poly(body, [[0, 0], [self.sensor_a/2, self.sensor_h],
                                                [-self.sensor_a/2, self.sensor_h]])
            sensor.sensor = True
            sensor.collision_type = self.SENSOR_COLLTYPE
            sensor.color = [85, 255, 115]

        self.space.add(body, trackL, trackR, sensor)
        h = self.space.add_collision_handler(self.SENSOR_COLLTYPE, self.TARGET_COLLTYPE)
        h.begin = self._sensor_activated
        h.separate = self._sensor_deactivated

        g = self.space.add_collision_handler(self.CREATURE_COLLTYPE, self.TARGET_COLLTYPE)
        g.begin = self._pick_up_target


        return body, trackL, trackR


    def _spawn_targets(self, area, radius):
        """
        Function that will spawn targets on given area.
        to be called manually.
        :param area: array of (x, y) coordinates of rectangular area origining at (0, 0)
        :param radius: radius of target
        """
        for i in xrange(self.num_of_targets):
            self._spawn_target(random.randint(0, area[0]), random.randint(0, area[1]), radius)


    def display_stuff(self):
        self.screen.fill([45, 54, 100])
        self.space.debug_draw(self.draw_options)
        if self.light_sources is not None:
            # Draw 'light source'. Note that pygame uses top left, while pymunk
            #   and therefore light_sources is in bottom left coordinates.
            for l_source in self.light_sources:
                x, y = l_source
                pygame.draw.circle(self.screen, (255, 220, 0),
                                   (x, self.HEIGHT-y), 15)
        pygame.display.flip()



    def mainloop(self):
        # Loop control variable
        running = True
        while running:
            for event in pygame.event.get():
                if (event.type == QUIT or
                        (event.type == KEYDOWN and event.key == K_ESCAPE)):
                    running = False
                 # Temporary steering - test everything
                if event.type == KEYDOWN and event.key == K_LEFT:
                    # Apply force
                    self.trackL_running = True
                    self.trackL.color = (255, 0, 0)

                elif event.type == KEYUP and event.key == K_LEFT:
                    # Disable that force
                    self.trackL_running = False
                    self.trackL.color = (255, 150, 25)

                if event.type == KEYDOWN and event.key == K_RIGHT:
                    self.trackR_running = True
                    self.trackR.color = (255, 0, 0)

                elif event.type == KEYUP and event.key == K_RIGHT:
                    self.trackR_running = False
                    self.trackR.color = (255, 150, 25)


            if self.trackL_running is True:
                self.creature.apply_impulse_at_local_point(self.engine_force, self.trackL_center)

            if self.trackR_running is True:
                self.creature.apply_impulse_at_local_point(self.engine_force, self.trackR_center)

            if not (self.trackL_running or self.trackR_running):
                self.creature.velocity = (0, 0)
                self.creature.angular_velocity = 0


            if self.score == self.num_of_targets:
                running = False


            self.space.step(1. / 50)

            if self.display is True:
                to_display = self.display_stuff()
                self.clock.tick(self.clock_time)

            self.timer += 1


    def step_sim(self, timestep=1/50.):
        # Reset pick up variable
        self.picked_up = False

        if self.trackL_running is True:
            self.creature.apply_impulse_at_local_point(self.engine_force, self.trackL_center)

        if self.trackR_running is True:
            self.creature.apply_impulse_at_local_point(self.engine_force, self.trackR_center)


        # Used in other variation of stepping
        # if self.trackL_stepping:
        #     self.creature.apply_impulse_at_local_point(self.engine_force, self.trackL_center)
        #
        # if self.trackR_stepping:
        #     self.creature.apply_impulse_at_local_point(self.engine_force, self.trackR_center)
        #


        self.space.step(timestep)

        if not (self.trackL_running or self.trackR_running) and not self.stepping:
            print "resetting velocity!"
            self.creature.velocity = (0, 0)
            self.creature.angular_velocity = 0

        # Used in other variation of stepping
        # if not (self.trackL_running or self.trackR_running or self.trackL_stepping or self.trackR_stepping):
        #     self.creature.velocity = (0, 0)
        #     self.creature.angular_velocity = 0


        # Measure distance from creature to the 'light source'
        # Note that we have two sonsory locations, correlating with track
        #   gravity centers.
        get_dist = lambda xl, yl, xc, yc: ((yl-yc)**2 + (xl-xc)**2)**0.5
        if self.light_sources is not None:
            self.light_distL = 0
            self.light_distR = 0
            sensorR = [self.trackL_center[0]-20, self.trackL_center[1]]
            sensorL = [self.trackR_center[0]+20, self.trackR_center[1]]

            print "measuring distance!"
            for l_source in self.light_sources:
                x, y = l_source
                self.light_distL += get_dist(x, y, *self.creature.local_to_world(sensorL))
                self.light_distR += get_dist(x, y, *self.creature.local_to_world(sensorR))


        if self.display is True:
            to_display = self.display_stuff()
            self.clock.tick(self.clock_time)


        self.timer += 1

        self.stepping = False
        # self.trackL_stepping = False
        # self.trackR_stepping = False


    def step_trackL(self, force=None):
        force = (0, force) if force is not None else self.engine_force
        # self.trackL_stepping = True  # <- used in other variation of stepping
        self.creature.apply_impulse_at_local_point(force, self.trackL_center)


    def step_trackR(self, force=None):
        force = (0, force) if force is not None else self.engine_force
        # self.trackR_stepping = True  # <- used in other variation of stepping
        self.creature.apply_impulse_at_local_point(force, self.trackR_center)


    def start_trackL(self):
        self.trackL_running = True


    def stop_trackL(self):
        self.trackL_running = False


    def start_trackR(self):
        self.trackR_running = True


    def stop_trackR(self):
        self.trackR_running = False


def dist2freq(dist):
    # Basically turn distance to ligth source to spiking frequency
    # This could be done by applying voltage to additional cell connected to
    #   sensory cell
    HEIGHT = 1080
    WIDTH = 1920
    freq_max = 150
    freq_min = 20
    max_dist = (HEIGHT**2 + WIDTH**2)**0.5
    return (dist - 0) * (freq_max - freq_min) / (max_dist - 0) + freq_min


if __name__ == "__main__":
    sim = Simulation(gravity=(0, 0), WIDTH=1920, HEIGHT=1080, damping=0,
                     engine_force=0.1, sensor_h=200, light_sources=[[100, 100]])
    # sim.mainloop()
    while sim.score != sim.num_of_targets:
        sim.step_sim(timestep=1/20.)

        movement = raw_input(">>> ")
        if movement == 'a':
            sim.step_trackL()
        elif movement == 'd':
            sim.step_trackR()
        elif movement == 'w':
            sim.step_trackR()
            sim.step_trackL()

        print "Picked up?", sim.picked_up
        print "Distance to light source: ", sim.light_distL, sim.light_distR

        # TEMP:
        # Now check distance to ligth source
        creature_sim_dt = 1./20

        distanceL = sim.light_distL
        distanceR = sim.light_distR
        Lcell_delay = 1./dist2freq(distanceL)
        Rcell_delay = 1./dist2freq(distanceR)

        print "freqs:", dist2freq(distanceL), dist2freq(distanceR)

        # NOTE: Neuron time resets each creature timestep, so creature_sim_dt is
        #       highest occuring neuron time value, but in ms.
        # Create stimulation dictionary
        stim_dict = {0: [Lcell_delay*i for i in xrange(int(creature_sim_dt/Lcell_delay))],
                     1: [Rcell_delay*i for i in xrange(int(creature_sim_dt/Rcell_delay))]}

        print stim_dict
        print "lens", len(stim_dict[0]), len(stim_dict[1])

    print "simulated time: ", sim.timer / 50.
