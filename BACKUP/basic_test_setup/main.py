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


dynDNA = basic_ga.static2dynamic_specimen('10000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010', basic_ga.architecture)

stimuli = {0: [5, 15, 50, 55, 70, 72, 74],
           1: [5, 15, 50, 55, 70, 72, 74]}

spec = ga.Specimen(dynDNA, Cells.MechanosensoryCell, graph=False)
# spec.fitness_score(stimuli=stimuli, reflex=100)
spec.construct_network()
spec.show_architecture()
spec.set_recording_vectors()
spec.apply_stimuli({0: [5, 15, 50, 55, 70, 72, 74],
                   1: [5, 15, 50, 55, 70, 72, 74]})
spec.setup_output_cells([5, 6])
# spec.graph_connections()
spec.run(100)
print spec.get_out_spikes()
spec.plot_cells_activity(mode=2)

raw_input("Press enter to exit")