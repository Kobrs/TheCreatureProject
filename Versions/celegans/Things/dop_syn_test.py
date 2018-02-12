import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt

import simrun
import Cells
import interactive
import ga
import basic_ga


plt.ion()

"""
Notes:
* This version uses one dimensional brain
"""

def create_syn(cell):
    # syn = h.dop_ExpSyn(cell.dend(0.5))
    syn = h.dop_ExpSynSTDP(cell.dend(0.5))
    # syn = h.ExpSyn(cell.dend(0.5))
    syn.tau = 2
    syn.dop = 100
    # syn.dopdr = 0.0002
    # syn.dopdr = 0.05
    syn.dopdr = 0.04
    cell.synlist.append(syn)


# TODO: The second milestone is pleasure system - dopamine.
cell = Cells.GenericCell()
# cell.create_synapses()
create_syn(cell)
# cell.synlist[-1].g = 1

# print cell.synlist[-1].g
# print cell.synlist[-1].tau
# print cell.synlist[-1].dopdr
rvec = h.Vector()
rvec.record(cell.synlist[-1]._ref_dop)



cell2 = Cells.GenericCell()
cell2.create_synapses(1)
cell2.dend.e_pas = -5
cell2_v_vec = h.Vector()
cell2_v_vec.record(cell2.soma(0.5)._ref_v)
# rvec.record(cell.soma(0.5)._ref_v)



nc = cell2.connect2target(cell.synlist[-1])
nc.weight[0] = 0.002
nc.threshold = -5

cell_w_vec = h.Vector()
cell_w_vec.record(cell.synlist[-1]._ref_g)


cell_v_vec = h.Vector()
cell_v_vec.record(cell.soma(0.5)._ref_v)



h.tstop = 50
h.run()

plt.figure(1)
plt.plot(rvec)
plt.plot(cell2_v_vec)
plt.plot(cell_v_vec)
plt.figure(2)
plt.plot(cell_w_vec)
plt.show()



raw_input('Press return to exit')
