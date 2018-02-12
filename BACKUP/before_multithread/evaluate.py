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

    stimuli = create_stimuli(input_cells, simdur, 2)
    print "Stimuli:", stimuli

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
