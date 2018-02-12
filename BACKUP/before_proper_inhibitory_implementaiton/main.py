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
stimuli = {0: [5, 15, 50, 55, 70, 72, 74],
           1: [5, 15, 50, 55, 70, 72, 74]}





def create_specimen():
    # Mutate basic thing which is preset in resonable range:
    DNA = '10000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010'
    DNA = "".join([('0' if b =='1' else '1') if np.random.rand() < 0.02 else b for b in DNA])

    # return [np.random.randint(0, 2) for _ in xrange(440)]
    return DNA



# FIXME: Write replacement for deap toolbox mutate function, which assumes that
#         specimen's DNA is a list of bools, not a bool string.


# FIXME: Fix inhibitory connections: THEY CANNOT HAVE NEGATIVE WEIGHTS!!!!!!!
#           For now I can use standard exp synapse with syn.e = -70,
#           but sometime later I want to explore receptors and transmitters
#           systems (look at the downloaded exapmples and mod files in main
#           NEURON direcory inside of my workplace)




def evaluate_specimen(DNA):
    # This funciton HAVE TO return tuple!
    DNA = "".join(map(lambda b: str(int(b)), DNA))

    # Since we're passing new interpreter DNA is our real genetic code, we
    #   don't have to convert it to the one used by GAinterpreter class.
    spec = ga.Specimen(DNA, Cells.MechanosensoryCell,
                       interpreter=basic_ga.interpreter)
    score = spec.fitness_score(stimuli=stimuli), 

    print "Evaluating specimen:"
    print DNA
    print spec.architecture
    print "Score:", score

    return score


# spec = ga.Specimen(dynDNA, Cells.MechanosensoryCell, graph=False)
# print spec.fitness_score(stimuli=stimuli, reflex=100)
# print spec.get_out_spikes()

# GA = ga.GA(mutate_rate=0.05, toursize=3, pop_size=25, CXPB=0.5, MUTPB=0.2)
# GA.create_functions(create_specimen, evaluate_specimen)
# GA.evolve()


DNA = "".join(map(str, create_specimen()))
# DNA = '10000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010'
# Evolved with score of 31853
# DNA = "11001101110011110110011111101111111011110111101110111111001011010110110111111111011111111011100111101111101110111100111101111110111111110111111111111110110011111110100111110011111100111101111001111101101111111110111001111100110101011111111101110011111111011111001111100011100111111111111010110011101011111011111101000110110111110000111111100011011101111111110101110010100011100000011011101011111101011111111000001011111101011110011111111001"
spec = ga.Specimen(DNA, Cells.MechanosensoryCell, interpreter=basic_ga.interpreter)
print spec.architecture
spec.construct_network()
spec.show_architecture()
spec.set_recording_vectors()
# spec.apply_stimuli({0: [5, 15, 50, 55, 70, 72, 74],
#                    1: [5, 15, 50, 55, 70, 72, 74]})
spec.setup_output_cells([5, 6])
# spec.graph_connections()
spec.run(100)
# print [nc.weight[0] for nc in spec.conn_list]
# for rvec in spec.record_vectors_dict:
#     print rvec, '\n\n\n'
print spec.get_out_spikes()
spec.plot_cells_activity(mode=2)



raw_input("Press enter to exit")