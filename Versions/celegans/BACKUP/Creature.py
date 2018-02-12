import numpy as np
from neuron import h
from matplotlib import pyplot as plt
from ascii_graph import Pyasciigraph

import time

import Cells
import Simulation
import ga


"""
Notes:
* This code is not that well optimized. For now the focus is to have properly
WORKING code, and only then make further efforts to optimize it.
"""


class Creature(object):
    def __init__(self, environment, creature_lifetime=0, sensor_cells=[1, 2, 3, 4, 5],
                 trackL_cells=[32, 33, 34, 35], trackR_cells=[64, 65, 66, 67],
                 engine_force=0.15, creature_spawn_loc=None,
                 motors_min_spikes=None, motors_max_spikes=None,
                 smooth_track_control=False, use_sensor=True, logging=False):
        """
        Class which creates creature - object containig both simulated body
        and neuron net to drive it.
        :param display_sim: decides whether use pygame to display creature sim
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


        self.environment = environment
        self.creature_sim_dt = environment.creature_sim_dt
        self.creature_lifetime = creature_lifetime*(1./self.creature_sim_dt)
        self.sensor_cells = sensor_cells
        # Currently we're just checkin if either of those cells spiked, but it may change
        self.trackL_cells = trackL_cells
        self.trackR_cells = trackR_cells
        self.logging = logging
        self.engine_force = engine_force
        self.motors_min_spikes = motors_min_spikes
        self.motors_max_spikes = motors_max_spikes
        self.WIDTH = environment.WIDTH
        self.HEIGHT = environment.HEIGHT

        self.num_of_regions = 3

        self.neuron2creature_dt = self.creature_sim_dt*1000  # basically convert s to ms
        self.movement_list = []
        self.alive = True
        self.plot_fig = None
        self.ascii_graph = Pyasciigraph()
        self.smooth_track_control = smooth_track_control

        quickest_run = 10
        self.body = Simulation.Body(
            environment=environment, engine_force=self.engine_force,
            sensor_h=200, sensor_a=40, creature_spawn_loc=creature_spawn_loc,
            use_sensor=use_sensor)

        self.max_score = float(self.environment.num_of_targets) / (quickest_run*(1./self.creature_sim_dt))


    def _init(self):
        # Network is constructed in Specmien's __init__ method
        # Setup output cells
        # self.net.setup_output_cells(self.trackL_cells + self.trackR_cells)
        # Actually setup all cells, so we can make statisctics
        self.net.setup_output_cells(self.net.cells_dict.keys())
        self.net.sim_init()

        # Kill disabled creatures (not able to live on their own)
        if not ((any(cell in self.net.cells_dict.keys() for cell in self.trackL_cells + self.trackR_cells)
                    and any(cell in self.net.cells_dict.keys() for cell in self.sensor_cells))):
            self._log("Killing disabled creature.")
            self.alive = False



    def _val_map(self, x, in_min, in_max, out_min, out_max):
        x = float(x)  # make sure that we get float operations
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


    def _log(self, val):
        if self.logging:
            print val


    def create_from_DNA(self, DNA):
        # Assuming that DNA is string of binary digits
        self.net = ga.Specimen(DNA_str, Cells.STDP_Dopamine_Cell,
                               num_of_regions=self.num_of_regions)

        self._init()  # finish initialization when we have network


    def create_from_architecture(self, architecture, dop, dopdr):
        interpreter = lambda _: (architecture, dop, dopdr)
        self.net = ga.Specimen("", Cells.STDP_Dopamine_Cell,
                               interpreter=interpreter,
                               num_of_regions=self.num_of_regions)
        self._init()  # finish initialization when we have network


    def stimulate_pleasure(self, region, dop=None, dopdr=None):
        self.net.release_dopamine(region, dop, dopdr)
        # self.net.sim_init()


    def apply_random_stim(self, n=5):
        stim_dict = dict([(np.random.randint(0,len(self.net.cells_dict.keys())), [1]) for _ in xrange(n)])
        self.net.apply_stimuli(stim_dict)

    def stimulate_cells(self, cells):
        stim_dict = dict([(c, 1) for c in cells])
        self.net.apply_stimuli(stim_dict)

    def apply_stimuli(self, stim_dict):
        # Direct wrapper for underlying ga object
        self.net.apply_stimuli(stim_dict)


    def add_stim_gate_cell(self, cell_id, targets):
        """
        This function creates cell used as gateway between sensory cell and
        direct current stimulation.
        :param cell_id: id of newly created gete cell. Note that they are stored
                        separately from main cells
        :param targets: list of targets ids.
        """
        self.net.add_stim_gate_cell(cell_id, targets)


    def stimulate_gate_cell(self, cell_id, amp, dur):
        """
        Stimulate cells added in add_stim_gate_cell with iclamp with provided
        current. Current should be in range 0.15 to 0.4
        """
        self.net.stimulate_gate_cell(cell_id, amp, dur)


    def get_position(self):
        return list(self.body.creature.position)


    def set_poistion(self, x, y):
        self.body.creature.position = [x, y]
        self.body.space.reindex_shapes_for_body(self.body.creature)


    def remove_light_source(self, i):
        self.environment.remove_light_source(i)


    def spawn_light_sources(self, n=1):
        self.environment.spawn_targets([self.WIDTH, self.HEIGHT], radius=15,
                                num_targets=n, light_source=True)


    def control_track_binary(self, spikes):
        trackL_activated = False
        trackR_activated = False
        hist_data = []
        for src_id, spikes in spikes.iteritems():
            # Check if it did spiked during our time window
            if len(spikes) > 0 and spikes[-1] > h.t-self.neuron2creature_dt:  # it cannot be > h.t
                if src_id in self.trackL_cells:
                    trackL_activated = True
                elif src_id in self.trackR_cells:
                    trackR_activated = True
                self.movement_list.append(self.body.timer)

            # Now we should make spikes plot
            if len(spikes) != 0:
                hist_data.append(("x=%d id=%d"%(self.net.cells_dict[src_id].x, src_id), len(spikes)))

        if trackL_activated:
            print "moving trackL"
            self.body.step_trackL()
        if trackR_activated:
            print "moving trackR"
            self.body.step_trackR()

        return hist_data


    def control_track_smooth(self, spikes):
        trackL_spikes = 0
        trackR_spikes = 0
        hist_data = []
        # NOTE: output spikes are reset each iteration, so they always contain
        #       only spikes from previous(current) timestep.
        for src_id, spikes in spikes.iteritems():
            trackL_spikes += len(spikes) if src_id in self.trackL_cells else 0
            trackR_spikes += len(spikes) if src_id in self.trackR_cells else 0

            # Now we should make spikes plot
            if len(spikes) != 0:
                hist_data.append(("x=%d id=%d"%(self.net.cells_dict[src_id].x, src_id), len(spikes)))

        trackL_force = self._val_map(trackL_spikes, self.motors_min_spikes, self.motors_max_spikes, 0, self.engine_force)
        trackR_force = self._val_map(trackR_spikes, self.motors_min_spikes, self.motors_max_spikes, 0, self.engine_force)
        self.body.step_trackL(trackL_force)
        self.body.step_trackR(trackR_force)

        if trackL_spikes > 0 or trackR_spikes > 0:
            self.movement_list.append(self.body.timer)

        return hist_data


    def step(self):
        # Check body's sensor state
        if self.body.detected:
            self._log("Sensor detected something!")
            stim_dict = {}
            for cell in self.sensor_cells:
                # Stimulate each sensor cell at the beginning of network sim
                stim_dict[cell] = [1]

            self.net.apply_stimuli(stim_dict)

            # NOTE: For now using net.run, which doesn't require this
            # Neccessary to append stimuli to NEURON simulation
            # self.net.sim_init()


        # Check if picked up something
        if self.body.picked_up:
            self._log("Picked up something and this makes me happy!")
            for i in xrange(self.num_of_regions-1 ): # 0 indexing
                self.net.release_dopamine(region=i)

            # NOTE: For now using net.run, which doesn't require this
            # Append changes to NEURON simulation
            # self.net.sim_init()


        # Get some feedback from our creature
        spikes = self.net.get_out_spikes()
        print "creature spike len=%d for creature"%np.size(spikes), self

        if self.smooth_track_control:
            hist_data = self.control_track_smooth(spikes)
        else:
            hist_data = self.control_track_binary(spikes)


        # Graph data
        for line in self.ascii_graph.graph('Spikes in last time window', hist_data):
            print line


        # Reset recording vectors (I'm not sure why it does that, but it does)
        self.net.sim_init()

        # Step NEURON simulation forward
        # self.net.advance(time=h.t+self.neuron2creature_dt)
        begin = time.clock()
        self.net.run(time=h.t+self.neuron2creature_dt)
        print "NEURON sim duration:", time.clock() - begin
        print "Simulated time:", h.t

        self.net.clear_sim()



        # Take care of kill variable
        if (len(self.movement_list) > 0 and self.movement_list[-1] <
              self.body.timer - (1*(1./self.creature_sim_dt))):
            # Kill after some time without activity (1s)
            self.alive = False

        # Kill after it's time's up or when collected all targets
        if not (self.body.timer < self.creature_lifetime and self.body.score < self.environment.num_of_targets):
            self.alive = False



    def setup_plotting(self, time_window=100):
        """
        Function used to dynamically plot cells potentials, but in practice
        it's too slow to be used.
        """

        self.net.set_recording_vectors(record_dend=False, record_t=False)
        # Prepare figure used for plotting

        self.plot_fig = plt.figure(16)
        plot_ax = self.plot_fig.add_subplot(111)

        self.lines = []
        x = np.arange(-time_window, 0, h.dt)
        y = np.ones(len(x))
        self.plot_points = len(x)

        for _ in xrange(len(self.net.cells_dict)):
            self.lines.append(plot_ax.plot(x, y)[0])



    def plot_activity(self):
        # update plot figure created earlier
        assert self.plot_fig is not None, "You haven't setup plotting!"

        # Get and plot the data
        i = 0
        for rvec in self.net.record_vectors_dict.values():
            y = rvec[1].as_numpy()[-self.plot_points:]  # we're only interested in soma_v_vec
            print np.shape(y)
            # print np.shape(rvec[1].as_numpy())
            # print rvec[1].as_numpy()
            # raise SystemExit
            print len(y), self.plot_points

            if len(y) == self.plot_points:
                self.lines[i].set_ydata(y)
                self.plot_fig.canvas.draw()

            i += 1




    def score(self):
        # If simulation was run, then score it
        if if_running:
            try:
                # Let's say that minimal time to finish the task is 10 seconds
                score = self.body.score / float(self.body.timer)
            except ZeroDivisionError:
                # This happens when creature was disabled and not tested at all
                score = 0
        else:
            # Else just give it score 0
            score = 0

        return self._val_map(score, 0, max_score, 0.1, 1.),  # as tuple
