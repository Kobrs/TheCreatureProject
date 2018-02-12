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
                 display=True):
        """
        Class which creates simulation, initializes world and so on.
        :param display: Specify whether initialize pygame to display simulation
        :param gravity: Gravity in pymunk
        :param sensor_h: height of triangle representing sensor field  of view
        :param sensor_a: base of triangle representing sensor field  of view
        :param iterations: Iterations for pymunk solver
        :param clock: sets value for pygame clock to maintain FPS
        """
        self.engine_force = (0, engine_force)
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.sensor_h = sensor_h
        self.sensor_a = sensor_a
        self.num_of_targets = num_of_targets
        self.display = display
        self.clock_time = clock

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
        # self.stepping = False
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
        body.position = [self.WIDTH/2, self.HEIGHT/2]
        # shape = pymunk.Poly.create_box(body, (10, 40))
        trackL = pymunk.Poly(body, [[-15, 20], [-15, -20], [-5, -20], [-5, 20]])
        trackR = pymunk.Poly(body, [[15, 20], [15, -20], [5, -20], [5, 20]])
        trackL.density = 0.000002
        trackR.density = 0.000002
        trackL.color = [75, 75, 75]
        trackR.color = [75, 75, 75]
        trackL.collision_type = self.CREATURE_COLLTYPE
        trackR.collision_type = self.CREATURE_COLLTYPE

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
                    self.trackL.color = (255, 0, 0, 255)

                elif event.type == KEYUP and event.key == K_LEFT:
                    # Disable that force
                    self.trackL_running = False
                    self.trackL.color = (255, 150, 25, 255)

                if event.type == KEYDOWN and event.key == K_RIGHT:
                    self.trackR_running = True
                    self.trackR.color = (255, 0, 0, 255)

                elif event.type == KEYUP and event.key == K_RIGHT:
                    self.trackR_running = False
                    self.trackR.color = (255, 150, 25, 255)


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

        # if not (self.trackL_running or self.trackR_running) and not self.stepping:
        #     print "resetting velocity!"
        #     self.creature.velocity = (0, 0)
        #     self.creature.angular_velocity = 0

        if self.trackL_stepping:
            self.creature.apply_impulse_at_local_point(self.engine_force, self.trackL_center)

        if self.trackR_stepping:
            self.creature.apply_impulse_at_local_point(self.engine_force, self.trackR_center)



        self.space.step(timestep)



        if not (self.trackL_running or self.trackR_running or self.trackL_stepping or self.trackR_stepping):
            self.creature.velocity = (0, 0)
            self.creature.angular_velocity = 0





        if self.display is True:
            to_display = self.display_stuff()

        self.clock.tick(self.clock_time)

        self.timer += 1
        # self.stepping = False
        self.trackL_stepping = False
        self.trackR_stepping = False


    def step_trackL(self):
        self.trackL_stepping = True


    def step_trackR(self):
        self.trackR_stepping = True


    def start_trackL(self):
        self.trackL_running = True


    def stop_trackL(self):
        self.trackL_running = False


    def start_trackR(self):
        self.trackR_running = True


    def stop_trackR(self):
        self.trackR_running = False



if __name__ == "__main__":
    sim = Simulation(gravity=(0, 0), WIDTH=1920, HEIGHT=1080, damping=0, engine_force=0.3, sensor_h=200)
    # sim.mainloop()
    while sim.score != sim.num_of_targets:
        movement = raw_input(">>> ")
        if movement == 'a':
            sim.step_trackL()
        elif movement == 'd':
            sim.step_trackR()
        elif movement == 'w':
            sim.step_trackR()
            sim.step_trackL()

        sim.step_sim(timestep=1/40.)
        print "Picked up?", sim.picked_up

    print "simulated time: ", sim.timer / 50.
