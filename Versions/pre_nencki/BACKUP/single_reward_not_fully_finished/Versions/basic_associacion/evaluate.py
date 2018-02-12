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
# stimuli = {0: [10, 12, 50, 55, 100, 103],
#            1: [25, 27, 70, 72, 74, 120, 122, 123]}
input_cells = [0, 1]
output_cells = [5, 6]
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


while True:
    DNA = raw_input("Please enter the desired Basic DNA >> ")
    plt.figure(1)
    plt.cla()
    plt.figure(2)
    plt.cla()
    plt.figure(3)
    plt.cla()
    DNA = DNA.strip()

    stimuli = create_stimuli(input_cells, simdur, 2)
    print "Stimuli:", stimuli

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
    print "SCORE:", spec.fitness_score(stimuli=stimuli, 
                                       output_cells=output_cells,
                                       interia=10, sim_time=simdur)
    timeline, mov_record, timers_record = spec.output_data


    fig = plt.figure(2)
    p1 = plt.plot(timeline, np.array(mov_record)[:, 0])
    p2 = plt.plot(timeline, np.array(timers_record)[:, 0])  # plot timers
    p3 = plt.plot(timeline, np.array(mov_record)[:, 1])
    p4 = plt.plot(timeline, np.array(timers_record)[:, 1])  # plot timers
    plt.legend([p1[0], p2[0], p3[0], p4[0]], ["mov0", "timer0", "mov1", "timer1"] )
    ax = fig.axes[0]
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(range(int(start), int(end), 5))

    fig = plt.figure(3)
    plt.plot(timeline, map(lambda a, b: a ^ b, *(np.array(mov_record).transpose())))
    ax = fig.axes[0]
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(range(int(start), int(end), 5))

    plt.show()
