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


input_cells = [0, 1]
output_cells = [5, 6, 7, 8]
simdur = 100


stimuli = {}

first = True
while True:
    DNA = raw_input("Please enter the desired Basic DNA >> ")
    plt.figure(1)
    plt.cla()
    plt.figure(2)
    plt.cla()
    plt.figure(3)
    plt.cla()
    # plt.figure(4)
    # plt.cla()
    DNA = DNA.strip()
    print DNA

    spec = ga.Specimen(DNA, Cells.MechanosensoryCell, graph=False)
    print spec.architecture
    spec.construct_network()
    spec.show_architecture()
    spec.set_recording_vectors()
    spec.apply_stimuli(stimuli, seed=1234)
    spec.setup_output_cells([5, 6])
    # spec.graph_connections()
    spec.run(100)
    print spec.get_out_spikes()
    spec.plot_cells_activity(mode=2)
    print "SCORE:", spec.fitness_score(stimuli=stimuli, sim_time=simdur,
                                       output_cells=output_cells, interia=10)
    timeline, mov_record, timers_record = spec.output_data


    print [(src, tgt, nc.weight[0]) for src, tgt, nc in spec.conn_list]


    plt.figure(2)
    p1 = plt.plot(timeline, np.array(mov_record)[:, 0])
    p2 = plt.plot(timeline, np.array(timers_record)[:, 0])  # plot timers
    p3 = plt.plot(timeline, np.array(mov_record)[:, 1])
    p4 = plt.plot(timeline, np.array(timers_record)[:, 1])  # plot timers
    plt.legend(p1+p2+p3+p4, ["mov_record[0]", "timers_record[0]", "mov_record[1]", "timers_record[1]"])
    plt.figure(3)
    p1 = plt.plot(timeline, np.array(mov_record)[:, 2])
    p2 = plt.plot(timeline, np.array(timers_record)[:, 2])  # plot timers
    p3 = plt.plot(timeline, np.array(mov_record)[:, 3])
    p4 = plt.plot(timeline, np.array(timers_record)[:, 3])  # plot timers
    plt.legend(p1+p2+p3+p4, ["mov_record[2]", "timers_record[2]", "mov_record[3]", "timers_record[3]"])

    # plt.figure(4)
    # plt.plot(timeline, map(lambda a, b: a and b, *(np.array(mov_record).transpose())))
    plt.show()
