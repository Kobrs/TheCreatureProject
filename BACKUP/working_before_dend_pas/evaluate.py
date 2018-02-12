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
stimuli = {0: [10, 12, 50, 55, 100, 103],
           1: [25, 27, 70, 72, 74, 120, 122, 123]}




    # DNA = "10000100110100000010100001001101000010111000010011110000001010000100111100000010100001001101000000101001010011010000001100000100110100000010100001001101000000101000010011010000001010000100110100000010100001011101000000101000010011010000001010001100110100000010110001001101000000100000010011010000001010100100110100000010100001001101000000101000010011010000001010000100110110000010100001001101000010101000010011010000001010000100110100000010"
while True:
    DNA = raw_input("Please enter the desired Basic DNA >> ")
    print DNA

    spec = ga.Specimen(DNA, Cells.MechanosensoryCell, interpreter=basic_ga.interpreter)
    print spec.architecture
    spec.construct_network()
    spec.show_architecture()
    spec.set_recording_vectors()
    spec.apply_stimuli(stimuli)
    spec.setup_output_cells([5, 6])
    # spec.graph_connections()
    spec.run(100)
    print spec.get_out_spikes()
    spec.plot_cells_activity(mode=2)
    spec.fitness_score(stimuli, th=1)
    t_vec, m_vec, if_mov = spec.output_data
    plt.figure()
    for i in xrange(len(t_vec)):
        plt.plot(t_vec[i], m_vec[i])
        plt.plot(t_vec[i], if_mov[i])
    plt.show()
