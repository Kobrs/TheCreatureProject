import numpy as np
from neuron import h
from matplotlib import pyplot as plt

import time

import simrun
import Cells
import Body_sim
import ga


"""
Notes:
* This code is not that well optimized. For now the focus is to have properly
WORKING code, and only then make further efforts to optimize it.
"""


class Creature(object):
    def __init__(self, display_sim=False, use_clock=False, creature_sim_dt=1./40,
                 creature_lifetime=0, sensor_cells=[1, 2, 3, 4, 5],
                 trackL_cells=[32, 33, 34, 35], trackR_cells=[64, 65, 66, 67],
                 num_of_targets=10, logging=False):
        """
        Class which creates creature - object containig both simulated body
        and neuron net to drive it.
        :param display_sim: decides whether use pygame to display creature sim
        :param use_clock: decides whether to use pygame clock to maintain steady framerate
        :param creature_sim_dt: timestep for creature simulation [s]
        :param creature_lifetime: time after which creature dies (used for GA)
                                  Used for self.alive parameter.
        :param sensor_cells: cells treated as sensory - stimulated from simulation
        :param trackL_cells: cells treated as motor - drive body's left track in simulation
        :param trackR_cells: cells treated as motor - drive body's right track in simulation
        :param num_of_targets: number of targets to be spawned in simulation
        :param logging: decides whether to log creature's activity from within this classs
        """


        self.creature_sim_dt = creature_sim_dt
        self.creature_lifetime = creature_lifetime*(1./creature_sim_dt)
        self.sensor_cells = sensor_cells
        # Currently we're just checkin if either of those cells spiked, but it may change
        self.trackL_cells = trackL_cells
        self.trackR_cells = trackR_cells
        self.logging = logging

        self.num_of_regions = 3

        self.neuron2creature_dt = creature_sim_dt*1000  # basically convert s to ms
        self.movement_list = []
        self.alive = True
        self.plot_fig = None

        quickest_run = 10
        self.body = Body_sim.Simulation(engine_force=10, sensor_h=200,
            sensor_a=40, num_of_targets=num_of_targets, WIDTH=1920, HEIGHT=1080,
            use_clock=use_clock, display=display_sim)

        self.max_score = float(num_of_targets) / (quickest_run*(1./creature_sim_dt))


    def _init(self):
        # Network is constructed in Specmien's __init__ method
        # Setup output cells
        self.net.setup_output_cells(self.trackL_cells + self.trackR_cells)
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


    def step(self):


        self.body.step_sim(timestep=self.creature_sim_dt)



        # TEMP
        # self.net.set_recording_vectors()




        # Check body's sensor state
        if self.body.detected: 
            self._log("Sensor detected something!")
            stim_dict = {}
            for cell in self.sensor_cells:
                # Stimulate each sensor cell at the beginning of network sim
                stim_dict[cell] = [0.025] 

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

        trackL_activated = False
        trackR_activated = False
        for src_id, spikes in spikes.iteritems():
            # Check if it did spiked during our time window
            if len(spikes) > 0 and spikes[-1] > h.t-self.neuron2creature_dt:  # it cannot be > h.t
                if src_id in self.trackL_cells:
                    trackL_activated = True
                elif src_id in self.trackR_cells:
                    trackR_activated = True
                self.movement_list.append(self.body.timer)                    

        if trackL_activated:
            print "moving trackL"
            self.body.step_trackL()
        if trackR_activated:
            print "moving trackR"
            self.body.step_trackR()



        # Reset recording vectors (?)
        self.net.sim_init()


        # Step NEURON simulation forward
        # self.net.advance(time=h.t+self.neuron2creature_dt)
        self.net.run(time=h.t+self.neuron2creature_dt)




        # TEMP
        # self.net.plot_cells_activity()
        # raw_input("asdf")
        # raise SystemExit





        # Take care of kill variable
        if (len(self.movement_list) > 0 and self.movement_list[-1] < 
              self.body.timer - (1*(1./self.creature_sim_dt))):
            # Kill after some time without activity (1s)
            self.alive = False

        # Kill after it's time's up or when collected all targets
        if not (self.body.timer < self.creature_lifetime and self.body.score < self.body.num_of_targets):
            self.alive = False



    def setup_plotting(self, time_window=100):
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

        import time




        # FIXME:
        #   IT turns out that using h.continuerun or/and h.init is quite
        #   dangerous by its own, as it tends to reset recording variable states,
        #   so I've to test it further and fix this code from this mistakes, as
        #   it's not looking like working one. (look into testing.py)





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
                time.sleep(2)

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
