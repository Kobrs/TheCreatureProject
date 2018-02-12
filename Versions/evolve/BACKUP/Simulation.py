import pygame
import pymunk
from pygame.locals import *
import pymunk.pygame_util
import collections
from copy import copy
from math import sqrt
import random

import Creature



class Environment(object):
    def __init__(self, num_of_targets, num_lsources, WIDTH, HEIGHT,
                 clock_time, display_sim, creature_sim_dt=1./20,
                 iterations=1000):
        """
        This class takes care of environment physics simulation. It will create
        space on which other creature bodies can be spawned.
        :param display_sim: Specify whether initialize pygame to display_sim simulation
        :param num_of_targets: number of targets to be spawn
        :param num_lsources: number of light sources to spawn
        :param iterations: Iterations for pymunk solver
        :param clock_time: sets value for pygame clock_time to maintain FPS
        :param creature_sim_dt: timestep for creature simulation [s]
        """

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.num_of_targets = num_of_targets
        self.display_sim = display_sim
        self.clock_time = clock_time
        self.creature_sim_dt = creature_sim_dt

        self.creatures = []  # this array will hold all existing creatures
        self.light_sources = []
        self.TARGET_COLLTYPE = 2

        # Needed only for displaying simulation
        if self.display_sim:
            pygame.init()
            self.screen = pygame.display.set_mode([WIDTH, HEIGHT])
            pygame.display.set_caption("specimen simulation")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.space =  pymunk.Space(threaded=True)
        self.space.threads = 8
        self.space.gravity = (0, 0)
        self.space.iterations = iterations


        self.spawn_targets([self.WIDTH, self.HEIGHT], num_of_targets, 10)
        self.spawn_targets([self.WIDTH, self.HEIGHT], num_lsources, 15, light_source=True)


    def _spawn_target(self, x, y, radius=10, light_source=False):
        body = pymunk.Body()
        body.position = [x, y]
        shape = pymunk.Circle(body, radius)
        shape.density = 1
        shape.collision_type = self.TARGET_COLLTYPE
        if light_source:
            shape.color = (255, 220, 0, 255)
        else:
            shape.color = (255, 70, 0, 255)

        self.space.add(body, shape)

        return body


    def _display_stuff(self):
        self.screen.fill([45, 54, 100])
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()


    def spawn_targets(self, area, num_targets, radius=10, light_source=False):
        """
        Function that will spawn targets on given area.
        to be called manually.
        :param area: array of (x, y) coordinates of rectangular area origining at (0, 0)
        :param radius: radius of target
        :param light_source: specify whether the target is light source, or not
        """
        for i in xrange(num_targets):
            x = random.randint(0, area[0])
            y = random.randint(0, area[1])
            body = self._spawn_target(x, y, radius, light_source)
            if light_source:
                self.light_sources.append(body)


    def remove_light_source(self, i):
        target = self.light_sources[i]
        # TODO: I guess that something like that shouldn't ever appear in here,
        #       but for now lets just ignore it...
        try:
            self.space.remove(target, *(target.shapes))
        except KeyError:
            pass
        del self.light_sources[i]


    def spawn_creature(
                 self, environment, creature_lifetime=0, sensor_cells=[1, 2, 3, 4, 5],
                 trackL_cells=[32, 33, 34, 35], trackR_cells=[64, 65, 66, 67],
                 engine_force=0.15, creature_spawn_loc=None, motors_min_spikes=None, motors_max_spikes=None,
                 smooth_track_control=False, use_sensor=True, logging=False):
        """
        Method to spawn new creature in current environment.
        :param creature_lifetime: time after which creature dies (used for GA)
                                  Used for self.alive parameter.
        :param sensor_cells: cells treated as sensory - stimulated from simulation
        :param trackL_cells: cells treated as motor - drive body's left track in simulation
        :param trackR_cells: cells treated as motor - drive body's right track in simulation
        :param motors_min_spikes: minimum numer of spikes of sensory cell,
                                   none if not using smooth_track_control
        :param motors_max_spikes: maximum numer of spikes of sensory cell,
                                   none if not using smooth_track_control
        :param smooth_track_control: set it if engine force should be dependend
                                     on number of motory cells
        :param logging: decides whether to log creature's activity from within this classs
        """
        creature = Creature.Creature(environment, creature_lifetime, sensor_cells, trackL_cells,
                          trackR_cells, engine_force, creature_spawn_loc, motors_min_spikes,
                          motors_max_spikes, smooth_track_control,
                          use_sensor, logging)
        self.creatures.append(creature)




    def step_sim(self):
        for creature in self.creatures:
            creature.step()
            body = creature.body
            # Reset pick up variable
            creature.picked_up = False

            if body.trackL_running is True:
                body.creature.apply_impulse_at_local_point(body.engine_force, body.trackL_center)

            if body.trackR_running is True:
                body.creature.apply_impulse_at_local_point(body.engine_force, body.trackR_center)


            # Used in other variation of stepping
            # if body.trackL_stepping:
            #     body.creature.apply_impulse_at_local_point(body.engine_force, body.trackL_center)
            #
            # if body.trackR_stepping:
            #     body.creature.apply_impulse_at_local_point(body.engine_force, body.trackR_center)
            #


        self.space.step(self.creature_sim_dt)

        for creature in self.creatures:
            body = creature.body
            if not (body.trackL_running or body.trackR_running) and not body.stepping:
                print "resetting velocity!"
                body.creature.velocity = (0, 0)
                body.creature.angular_velocity = 0

            # Used in other variation of stepping
            # if not (body.trackL_running or body.trackR_running or body.trackL_stepping or body.trackR_stepping):
            #     body.creature.velocity = (0, 0)
            #     body.creature.angular_velocity = 0


            # Measure distance from creature to the 'light source'
            # Note that we have two sonsory locations, correlating with track
            #   gravity centers.
            get_dist = lambda xl, yl, xc, yc: ((yl-yc)**2 + (xl-xc)**2)**0.5
            body.light_distL = 0
            body.light_distR = 0
            sensorR = [body.trackL_center[0]-20, body.trackL_center[1]]
            sensorL = [body.trackR_center[0]+20, body.trackR_center[1]]

            print "measuring distance!"
            for l_source in self.light_sources:
                x, y = l_source.position
                body.light_distL += get_dist(x, y, *body.creature.local_to_world(sensorL))
                body.light_distR += get_dist(x, y, *body.creature.local_to_world(sensorR))

            body.timer += 1
            body.stepping = False
            # body.trackL_stepping = False
            # body.trackR_stepping = False

        if self.display_sim is True:
            to_display = self._display_stuff()
            self.clock.tick(self.clock_time)



class Body(object):
    def __init__(self, environment, engine_force=0.1, sensor_h=80, sensor_a=40,
                 creature_spawn_loc=None, use_sensor=True):
        """
        Class which creates simulation, initializes world and so on.
        :param environment: Environment object on which this creature will live
        :param gravity: Gravity in pymunk
        :param sensor_h: height of triangle representing sensor field  of view
        :param sensor_a: base of triangle representing sensor field  of view
        :param use_sensor: decides whether to create basic binary collistion sensor.
        """
        self.engine_force = (0, engine_force)

        self.WIDTH = environment.WIDTH
        self.HEIGHT = environment.HEIGHT

        self.sensor_h = sensor_h
        self.sensor_a = sensor_a

        self.use_sensor = use_sensor
        self.creature_spawn_loc = creature_spawn_loc if creature_spawn_loc is not None else [WIDTH/2, HEIGHT/2]

        self.SENSOR_COLLTYPE = 1
        self.TARGET_COLLTYPE = 2
        self.CREATURE_COLLTYPE = 3
        self.space = environment.space

        self.sensor_detected = False
        self.score = 0


        # Initialize simulation
        self.creature, self.trackL, self.trackR =  self.spawn_creature()
        self.trackL_center = [self.trackL.center_of_gravity.x, self.trackL.center_of_gravity.y]
        self.trackR_center = [self.trackR.center_of_gravity.x, self.trackR.center_of_gravity.y]

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



    def spawn_creature(self):
        body = pymunk.Body()
        body.position = self.creature_spawn_loc
        # shape = pymunk.Poly.create_box(body, (10, 40))
        trackL = pymunk.Poly(body, [[-15, 20], [-15, -20], [-5, -20], [-5, 20]])
        trackR = pymunk.Poly(body, [[15, 20], [15, -20], [5, -20], [5, 20]])
        trackL.density = 0.000002
        trackR.density = 0.000002
        trackL.color = (75, 75, 75, 255)
        trackR.color = (75, 75, 75, 255)
        trackL.collision_type = self.CREATURE_COLLTYPE
        trackR.collision_type = self.CREATURE_COLLTYPE

        if self.use_sensor:
            # Add sesnor
            sensor = pymunk.Poly(body, [[0, 0], [self.sensor_a/2, self.sensor_h],
                                                [-self.sensor_a/2, self.sensor_h]])
            sensor.sensor = True
            sensor.collision_type = self.SENSOR_COLLTYPE
            sensor.color = (85, 255, 115, 255)
            self.space.add(sensor)

        self.space.add(body, trackL, trackR)
        h = self.space.add_collision_handler(self.SENSOR_COLLTYPE, self.TARGET_COLLTYPE)
        h.begin = self._sensor_activated
        h.separate = self._sensor_deactivated

        g = self.space.add_collision_handler(self.CREATURE_COLLTYPE, self.TARGET_COLLTYPE)
        g.begin = self._pick_up_target


        return body, trackL, trackR





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
