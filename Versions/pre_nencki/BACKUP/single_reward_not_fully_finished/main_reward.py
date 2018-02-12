import numpy as np
# from neuron import h, gui
from neuron import h
from matplotlib import pyplot as plt
from deap import base
# from deap import creator
from deap import tools

import time

import simrun
import Cells
import interactive
import ga
import Body_sim

"""
Notes:
* This version uses one dimensional brain
"""

display_sim = False
creature_sim_dt = 1/40.  # s
creature_lifetime = 15*(1/creature_sim_dt)  # first number is in seconds
sensor_cells = [1, 2, 3, 4, 5]
# Currently we're just checkin if either of those cells spiked, but it may change
trackL_cells = [32, 33, 34, 35]
trackR_cells = [64, 65, 66, 67]

neuron2creature_dt = creature_sim_dt*1000  # basically convert s to ms

np.random.seed(1324)
plt.ion()


def val_map(x, in_min, in_max, out_min, out_max):
  x = float(x)  # make sure that we get float operations
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def create_specimen(len_min=100, len_max=20000):
    return [bool(np.random.randint(0, 2)) for _ in xrange(np.random.randint(len_min, len_max))]



def evaluate_specimen(DNA, generation=0):
    # NOTE: I may comment this assertion before real evolution, but it's
    #   helpful for debugging.
    for bit in DNA:
        assert(type(bit) is bool), "DNA have to be list of bools"

    # Now we can convert it to this form, which is used by GAinterpreter
    DNA_str = "".join(map(lambda b: str(int(b)), DNA))

    # So now begins the fun part: stitch together creature simulation and network
    creature = Body_sim.Simulation(engine_force=10, sensor_h=200,
        sensor_a=40, num_of_targets=10, WIDTH=1920, HEIGHT=1080,
        use_clock=False, display=display_sim)
    net = ga.Specimen(DNA_str, Cells.DopamineCell, num_of_regions=3)
    # Network is constructed in Specmien's __init__ method
    # Setup output cells
    net.setup_output_cells(trackL_cells + trackR_cells)
    net.sim_init()

    movemement_list = []


    print "evaluate"

    # Decide whether to run this simulation at all
    if_running = False
    if (any(cell in net.cells_dict.keys() for cell in trackL_cells + trackR_cells)
        and any(cell in net.cells_dict.keys() for cell in sensor_cells)):
        if_running = True

    while creature.timer < creature_lifetime and creature.score < creature.num_of_targets and if_running:
        # Step creature simulation
        creature.step_sim(timestep=creature_sim_dt)

        if creature.detected:  # check creature's sensor state
            print "Sensor detected stimulation and sim init"
            # Give feedback to net simulation by stimulating sensory cells
            stim_dict = {}
            for cell in sensor_cells:
                # Give each cell single stimuli at first tiemstep
                stim_dict[cell] = [0.025]

            net.apply_stimuli(stim_dict)
            # To append this stimuli to NEURON simulation we have to initialize simulation
            net.sim_init()

        if creature.picked_up:
            print "Picked up dopamine release and sim init"
            # Reward it with pleasure stimuli
            for i in xrange(2):
                # release dopamine, but not into undefined region
                net.release_dopamine(region=i)
                # To make changes visible to NEURON simulation we have to call it
                net.sim_init()

        # Decide if we should move our creature
        spikes = net.get_out_spikes()

        trackL_activated = False
        trackR_activated = False
        for src_id, spikes in spikes.iteritems():
            # Check if it did spiked during our time window
            if len(spikes) > 0 and spikes[-1] > h.t-neuron2creature_dt:  # it cannot be > h.t
                if src_id in trackL_cells:
                    trackL_activated = True
                elif src_id in trackR_cells:
                    trackR_activated = True
                movemement_list.append(creature.timer)                    

        if trackL_activated:
            print "moving trackL"
            creature.step_trackL()
        if trackR_activated:
            print "moving trackR"
            creature.step_trackR()



        # Kill after some time without activity (1s)
        if len(movemement_list) > 0 and movemement_list[-1] < creature.timer - (1*(1./creature_sim_dt)):
            # Avoid both running net and being normally scored
            if_running = False
            break



        # TODO: THAT'S NICE AND SO ON, BUT I'VE FORGOT ABOUT STDP...




        # Step nueron simulation appropriate time forward
        # net.run(time=h.t+neuron2creature_dt)
        net.advance(time=h.t+neuron2creature_dt)


    max_score = float(creature.num_of_targets) / (10*(1/creature_sim_dt))
    # If simulation was run, then score it
    if if_running:
        # Let's say that minimal time to finish the task is 10 seconds
        score = creature.score / float(creature.timer)
    else:
        # Else just give it score 0
        score = 0

    # Scale score so it's within 0.1-1. range. 0.1 is to ensure score > 0 for roulette selection
    return val_map(score, 0, max_score, 0.1, 1.),


creator4scoop = ga.scoop_creator_init()

if __name__ == "__main__":
    print "LET'S EVOLVE THIS SHIT!"
    GA = ga.GA(mutate_rate=0.1, toursize=10, pop_size=100, CX_max_len_diff=10000, MUTPB=0.8, DMPB=0.5, TRPB=0.8, creator=creator4scoop, scoop=True)
    GA.create_functions(create_specimen, evaluate_specimen)
    GA.evolve()





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
