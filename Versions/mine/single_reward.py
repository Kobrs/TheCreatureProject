import numpy as np
from neuron import h
from matplotlib import pyplot as plt

import time

import Cells
import interactive
import ga
import Creature
import ControlGUI

"""
Notes:
* This version uses one dimensional brain
"""

display_sim = True
creature_sim_dt = 1/40.  # s
creature_lifetime = 60*(1/creature_sim_dt)  # first number is in seconds
sensor_cells = [1, 2, 3, 4, 5]
# Currently we're just checkin if either of those cells spiked, but it may change
trackL_cells = [32, 33, 34, 35]
trackR_cells = [64, 65, 66, 67]
num_of_regions = 10  # Number of brain regions. It doesn't have to reasemble brain fucntions

neuron2creature_dt = creature_sim_dt*1000  # basically convert s to ms

np.random.seed(8880)
# seed = np.random.randint(0,10000)
# print "Using the follwoing random generator seed:", seed
# np.random.seed(seed)
plt.ion()



def create_specimen(len_min=100, len_max=20000):
    return [bool(np.random.randint(0, 2)) for _ in xrange(np.random.randint(len_min, len_max))]


# Create creature's genome
DNA = create_specimen(len_min=10000, len_max=500000)
# Convert this into string of binary digits
DNA_str = "".join(map(lambda b: str(int(b)), DNA))
interpreter = ga.GAinterpreter()
interpreter.id_len = 10  # So now we have max of 1024 cells
architecture, dop, dopdr = interpreter.decode(DNA_str)



conf = dict([("dopamine-r%d"%i, False) for i in xrange(num_of_regions)])
conf["random_stim"] = False
conf["random_stim_cells"] = 50.
conf["sensor_stim"] = False
im = ControlGUI.ControlPanel(conf)


# Initailize our creature!
creature = Creature.Creature(display_sim=display_sim, clock=0,
    creature_sim_dt=creature_sim_dt, sensor_cells=sensor_cells,
    trackL_cells=trackL_cells, trackR_cells=trackR_cells,
    creature_lifetime=creature_lifetime, num_of_targets=0, logging=True)
creature.create_from_architecture(architecture, dop, dopdr)

while True:
    print "stepping"
    creature.step()

    # Loop for all regions
    for i in xrange(num_of_regions):
        if im.conf['dopamine-r%d'%i] is True:
            print "Feel the pleasure"
            # Stimulate and reset button
            creature.stimulate_pleasure(region=i, dop=100, dopdr=0.005)  # those setting result in quite nice time of dopamine action
            im.set_state("dopamine-r%d"%i, False)

    if im.conf["random_stim"]:
        print "Applying random stimulation!"
        creature.apply_random_stim(n=int(im.conf["random_stim_cells"]))
        # im.set_state("random_stim", False)
        # add dop and dopdr config to the gui - float support


    if im.conf["sensor_stim"]:
        print "Stimulating sensor cells!"
        creature.stimulate_cells(sensor_cells)


    # Another thing to do is making conditioning attepmt automatic - instead of doing it through
    #   real time interface with preview, I should be able to cope with this task by writing
    #   code which will do the same. Also its important that if we concider Creature, the timestep
    #   is inheretly simulation timestep, so we shouldn't mess with in between timesteps timing
    #   modification to induce conditioning, we should rather stick to condtions in practice, so
    #   the one where only creature timestep is the one from simulation.



    # Firstly we have to divide creature lifetime into two phases:
    #   * Training phase
    #   * Test phase





# import timeit
# sim_dt = creature_sim_dt

# DNA_str = "101101010101111011111000010011110000000001000100001000000001001000000000001010000100110100011001111"
# net = ga.Specimen(DNA_str, Cells.DopamineCell, num_of_regions=3)
# net.construct_network()
# print net.architecture
# print net.cells_dict
# net.set_recording_vectors()
# net.apply_stimuli({1: [sim_dt*400*i for i in xrange(10)]})

# # Note that this have to be called BEFORE apply_stimuli and set_recording_vectors
# net.sim_init()

# def run():
#     for i in xrange(10000):
#         net.advance(time=h.t+sim_dt)
#         # net.run(time=h.t+sim_dt)

# def run_():
#     net.run(time=h.t+10000*sim_dt)

# # t1 = timeit.timeit(run, number=2)
# # t2 = timeit.timeit(run_, number=2)
# # print "We get following timings:", t1, t2, "t1 is", t1/t2, "times bigger than t2."

# # net.run()
# run()
# # run_()
# net.plot_cells_activity(set_scale=False)

# raw_input("Enter to contunue!")

# run()
# net.plot_cells_activity(set_scale=False)

# raw_input('Press return to exit')
