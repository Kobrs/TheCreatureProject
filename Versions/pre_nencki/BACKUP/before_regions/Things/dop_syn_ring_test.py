import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt
from deap import base
from deap import creator
from deap import tools

import simrun
import Cells
import interactive
import ga
import basic_ga


np.random.seed(1324)
plt.ion()


# TODO: Change synapses to dop_expsyn and add their free parameters to DNA



"""
Notes:
* This version uses one dimensional brain
"""


cells = []
N = 10
r = 50  # Radius of cell locations from (0, 0, 0) in microns
for i in xrange(N):
    cell = Cells.DopamineCell()
    # When cells are created, the soma location is at (0,0,0) and
    # the dendrite extends along the x-axis.
    # First, at the origin, rotate about Z
    cell.rotateZ(i*2*np.pi/N)
    # Then reposition
    x_loc = np.cos(i*2*np.pi/N)*r
    y_loc = np.sin(i*2*np.pi/N)*r
    cell.set_position(x_loc, y_loc, 0)
    cells.append(cell)


shape_window = h.PlotShape()
shape_window.exec_menu("Show Diam")

stim = h.NetStim()  # Make a spike generator

# Attach it to a synapse in the middle of the dendrite of the first cell int
#   the network. (Named 'syn_' to avoid being overwritten wiht the syn var
#   used later.)
# syn_ = h.ExpSyn(cells[0].dend(0.5), name='syn_')
syn_ = h.ExpSyn(cells[0].dend(0.5))

stim.number = 1
stim.start = 9
ncstim = h.NetCon(stim, syn_)
ncstim.delay = 1
ncstim.weight[0] = 0.04  # NetCon weight is a vector
syn_.tau = 2


# Connect the cells
nclist = []
syns = []
for i in xrange(N):
    src = cells[i]
    tgt = cells[(i+1)%N]
    # syn = h.ExpSyn(tgt.dend(0.5))
    # syns.append(syn)
    # nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
    cells[(i+1)%N].create_synapses(1)
    tgt_syn = cells[(i+1)%N].synlist[-1]
    nc = src.connect2target(tgt_syn)

    nc.weight[0] = 0.002
    nc.delay = 5
    nclist.append(nc)


# Stimulate synapses with dopamine
for cell in cells:
    cell.dopamine_reaction()


# record dop from one, arbitrary synapse(all of them bahave exactly the same
#   except the one for spike source)
dop_vec = h.Vector()
dop_vec.record(tgt_syn._ref_dop)


# Current recording variables
syn_i_vec = h.Vector()
syn_i_vec.record(syn_._ref_i)

soma_v_vec_list = []
dend_v_vec_list = []
for cell in cells:
    soma_v_vec, dend_v_vec, t_vec = simrun.set_recording_vectors(cell)
    soma_v_vec_list.append(soma_v_vec)
    dend_v_vec_list.append(dend_v_vec)

t_vec_nc = h.Vector()
id_vec = h.Vector()
for i in range(len(nclist)):
    nclist[i].record(t_vec_nc, id_vec, i)

simrun.simulate(100)


fig = plt.figure(figsize=(8,4)) # Default figsize is (8,6)
for i in xrange(len(cells)):
    simrun.show_output(soma_v_vec_list[i], dend_v_vec_list[i], t_vec, fig,
                       mode=2)

plt.figure(2)
dop_plot = plt.plot(t_vec, dop_vec)
plt.legend(dop_plot, ["dopamine level"])



raw_input('Press return to exit')



# This is the DopamineCell class used:
# class DopamineCell(GenericCell):
#     def dopamine_reaction(self):
#         """This function is activated when cell is under influence of dopamine.
#         It is responsble for applying any changes in activity related to it
#         """

#         # TEMP - only for testing dopamine mechanisms
#         for syn in self.synlist:
#             syn.dop = 0.3
#             # syn.dopdr = -0.00002
#             syn.dopdr = 0.00002


#     def create_synapses(self, n=1):
#         """Add an exponentially decaying synapse in the middle
#         of the dendrite. Set its tau to 2ms, and append this
#         synapse to the synlist of the cell."""
#         for i in xrange(n):
#             syn = h.dop_ExpSyn(self.dend(0.5))
#             # syn = h.ExpSyn(self.dend(round(np.random.rand(), 2)))
#             syn.tau = 2
#             self.synlist.append(syn) # synlist is defined in Cell

