# NOTE: Basic implementation (look into notes for definiton)

import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt
from deap import base
from deap import creator
from deap import tools

import simrun
import Cells
import interactive as i
import ga
import basic_ga

plt.ion()


# TODO: stimuli should be variable!
# stimuli = {0: [5, 15, 50, 55, 70, 72, 74],
#            1: [5, 15, 50, 55, 70, 72, 74]}
# stimuli = {0: [10, 12, 50, 55, 100, 103],
#            1: [25, 27, 70, 72, 74, 120, 122, 123]}
stimuli = {}


input_cells = [0, 1]
output_cells = [5, 6]
simdur = 300


def randomize_spikes(simdur=100, spikes=2):
    """Create variable simuli with two randomly placed spikes but no closer to
    each other than 20ms and at least 10ms from both beginning and end.
    :param simdur: duration of simulation to be used"""

    times = []
    # Simplification: we always set spikes on natural numbers
    ok = True
    while len(times) < spikes:
        sp = np.random.randint(10, simdur-10)
        for t in times:
            if abs(t-sp) < 20:
                ok = False
        if ok is True:
            times.append(sp)
        ok = True

    return sorted(times)  # we have to put them in order!


def create_stimuli(input_cells, simdur=100, spikes=2):
    """Code for generating stimuli dictionary randomly according to the rules
    specified in randomize_spikes.
    :param input_cells: cells on whose stimuli should be aplied.
    :param simdur: duration of simulation
    :param spikes: number of spikes to apply"""
    stimuli = {}
    for cell in input_cells:
        stimuli[cell] = randomize_spikes(simdur, spikes)

    return stimuli


def create_specimen():
    # Mutate basic thing which is preset in resonable range:
    # DNA = '10000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010'
    # DNA = "".join([('0' if b =='1' else '1') if np.random.rand() < 0.02 else b for b in DNA])

    return "".join([str(np.random.randint(0, 2)) for _ in xrange(504)])
    # return DNA



def evaluate_specimen(DNA, generation=0):
    # This funciton HAVE TO return tuple!
    DNA = "".join(map(lambda b: str(int(b)), DNA))


    # Since we're passing new interpreter DNA is our real genetic code, we
    #   don't have to convert it to the one used by GAinterpreter class.

    # Prepare first stimuli
    stimuli1 = create_stimuli(input_cells, simdur=120, spikes=np.random.randint(1, 4))
    spec = ga.Specimen(DNA, Cells.MechanosensoryCell,
                       interpreter=basic_ga.interpreter)
    score1 = spec.fitness_score(stimuli=stimuli1, output_cells=output_cells,
                               interia=10, sim_time=simdur), 
    # Prepare second stimuli
    # stimuli2 = create_stimuli(input_cells, simdur=120, spikes=np.random.randint(1, 4))
    # score2 = spec.fitness_score(stimuli=stimuli2, output_cells=output_cells,
    #                            interia=10, sim_time=simdur), 

    # Encourage more consistant score
    # score = np.mean([score1, score2])
    score = score1


    if generation % 2 ==0:
        print "Evaluating specimen:"
        # print "Stimuli:", stimuli1, stimuli2
        print "Stimuli:", stimuli1
        print DNA
        print spec.architecture
        print "Score:", score
        print ""

    # There is significant slowdown and increase in RAM usage, check if this
    #   helps.
    del spec

    return score 


# spec = ga.Specimen(dynDNA, Cells.MechanosensoryCell, graph=False)
# print spec.fitness_score(stimuli=stimuli, reflex=100)
# print spec.get_out_spikes()

GA = ga.GA(mutate_rate=0.1, toursize=6, pop_size=50, CXPB=0.5, MUTPB=0.8)
GA.create_functions(create_specimen, evaluate_specimen)
GA.evolve()

# FIXME: Somewhere we're acquiring more data as program runs therefore slowing
#       it down with time of evolution




# DNA = "".join(map(str, create_specimen()))
# # # Score: 28
# # DNA = "10000100110100000010100001001101000010111000010011110000001010000100111100000010100001001101000000101001010011010000001100000100110100000010100001001101000000101000010011010000001010000100110100000010100001011101000000101000010011010000001010001100110100000010110001001101000000100000010011010000001010100100110100000010100001001101000000101000010011010000001010000100110110000010100001001101000010101000010011010000001010000100110100000010"
# spec = ga.Specimen(DNA, Cells.MechanosensoryCell, interpreter=basic_ga.interpreter)
# # print spec.architecture
# # spec.construct_network()
# # spec.show_architecture()
# # spec.set_recording_vectors()
# # spec.apply_stimuli(stimuli)
# # spec.setup_output_cells([5, 6])
# # # spec.graph_connections()
# # spec.run(100)
# # print spec.get_out_spikes()
# # spec.plot_cells_activity(mode=2)
# spec.fitness_score(stimuli)
# # t_vec, m_vec, if_mov = spec.output_data
# # plt.figure()
# # for i in xrange(len(t_vec)):
# #     plt.plot(t_vec[i], m_vec[i])
# #     plt.plot(t_vec[i], if_mov[i])
# plt.show()




raw_input("Press enter to exit")