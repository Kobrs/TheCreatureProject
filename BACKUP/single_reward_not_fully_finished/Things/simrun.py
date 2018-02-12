from neuron import h
from matplotlib import pyplot as plt

def set_recording_vectors(cell):
    """Set soma, dendrite, and time recording vectors on the cell.

    :param cell: Cell to record from.
    :return: the soma, dendrite, and time vectors as a tuple.
    """
    soma_v_vec = h.Vector()   # Membrane potential vector at soma
    dend_v_vec = h.Vector()   # Membrane potential vector at dendrite
    t_vec = h.Vector()        # Time stamp vector
    soma_v_vec.record(cell.soma(0.5)._ref_v)
    dend_v_vec.record(cell.dend(0.5)._ref_v)
    t_vec.record(h._ref_t)

    return soma_v_vec, dend_v_vec, t_vec

def simulate(tstop=25):
    """Initialize and run a simulation.

    :param tstop: Duration of the simulation.
    """
    h.tstop = tstop
    h.run()

def show_output(soma_v_vec, dend_v_vec, t_vec, fig, mode=0):
    """Draw the output.

    :param soma_v_vec: Membrane potential vector at the soma.
    :param dend_v_vec: Membrane potential vector at the dendrite.
    :param t_vec: Timestamp vector.
    :param new_fig: Flag to create a new figure (and not draw on top
            of previous results)
    :param mode: 0 for normal with black and red colors,
                 1 for multicorlor with dend
                 2 for multicolor without dend
    """
    if mode == 0:
        soma_plot = plt.plot(t_vec, soma_v_vec, color='black')
        dend_plot = plt.plot(t_vec, dend_v_vec, color='red')
    elif mode == 1:
        soma_plot = plt.plot(t_vec, soma_v_vec)
        dend_plot = plt.plot(t_vec, dend_v_vec)
    elif mode == 2:
        soma_plot = plt.plot(t_vec, soma_v_vec)
    else:
        print "[ERROR] Wrong plotting mode!"        
        raise SystemExit

    ax = fig.axes[0]
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(range(int(start), int(end), 5))

    plt.legend(soma_plot + dend_plot, ['soma', 'dend(0.5)'])
    plt.xlabel('time (ms)')
    plt.ylabel('mV')