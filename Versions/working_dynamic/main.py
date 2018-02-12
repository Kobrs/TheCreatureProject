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
output_cells = [5, 6, 7, 8]
simdur = 100


def create_specimen(len_min=100, len_max=2000):
    # Mutate basic thing which is preset in resonable range:
    # DNA = '10000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010'
    # DNA = "".join([('0' if b =='1' else '1') if np.random.rand() < 0.02 else b for b in DNA])

    # return "".join([str(np.random.randint(0, 2)) for _ in xrange(504)])
    return [bool(np.random.randint(0, 2)) for _ in xrange(np.random.randint(len_min, len_max))]
    # return DNA



def evaluate_specimen(DNA, generation=0):
    # This funciton HAVE TO return tuple!
    for bit in DNA:
        assert(type(bit) is bool), "DNA have to be list of bools"
    # Now we can convert it to this form, which is used by GAinterpreter
    DNA_str = "".join(map(lambda b: str(int(b)), DNA))

    spec = ga.Specimen(DNA_str, Cells.MechanosensoryCell)
    score = spec.fitness_score(stimuli=stimuli, output_cells=output_cells,
                               interia=10, sim_time=simdur), 
    # score = spec.fitness_score(stimuli={}, output_cells=output_cells,
    #                            interia=10), 

    if generation % 2 ==0:
        print "Evaluating specimen:"
        print "Stimuli:", stimuli
        print DNA_str
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

GA = ga.GA(mutate_rate=0.1, toursize=10, pop_size=50, CXPB=0.5, MUTPB=0.8, DMPB=0.5, TRPB=0.8)
GA.create_functions(create_specimen, evaluate_specimen)
GA.evolve()


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




# raw_input("Press enter to exit")