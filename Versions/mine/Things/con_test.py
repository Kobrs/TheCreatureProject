import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt

import Cells
import interactive


plt.ion()

"""
Notes:
* This version uses one dimensional brain

Connects cell1 

"""


def create_syn(cell):
    # Creates basic synapse used to connect stimuli
    syn = h.ExpSyn(cell.dend(0.5))
    syn.tau = 2
    # Return it instead of adding to cell's synlist
    return syn


# Create cells
cell1 = Cells.GenericCell()
cell2 = Cells.GenericCell()


# Add stimuli to both cells
ssyn1 = create_syn(cell1)
sns1 = h.NetStim()
sns1.interval = 20
sns1.number = 3
sns1.start = 10
snc1 = h.NetCon(sns1, ssyn1)
snc1.weight[0] = 0.04

# ssyn2 = create_syn(cell2)
# sns2 = h.NetStim()
# sns2.interval = 20
# sns2.number = 3
# sns2.start = 5
# snc2 = h.NetCon(sns2, ssyn2)
# snc2.weight[0] = 0.04


# Addditional stimuli to prevent regularity
ssyn3 = create_syn(cell1)
sns3 = h.NetStim()
sns3.interval = 50
sns3.number = 3
sns3.start = 200
snc3 = h.NetCon(sns3, ssyn3)
snc3.weight[0] = 0.04



# Connect both cells with STDP synapse
cell1.create_synapses(1)  # cell1 is presynaptic
nc = cell2.connect2target(cell1.synlist[-1])  # cell2 is postsynaptic
nc.weight[0] = 0.002



# Setup recording vectors
cell1_v_vec = h.Vector()
cell1_v_vec.record(cell1.soma(0.5)._ref_v)

cell2_v_vec = h.Vector()
cell2_v_vec.record(cell2.soma(0.5)._ref_v)

# cell1_w_vec = h.Vector()
# cell1_w_vec.record(cell1.synlist[-1]._ref_g)

tvec = h.Vector()
tvec.record(h._ref_t)



h.tstop = 500
h.run()

plt.figure(1)
plt.plot(tvec, cell1_v_vec)
plt.plot(tvec, cell2_v_vec)
# plt.figure(2)
# plt.plot(tvec, cell1_w_vec)
plt.show()



raw_input('Press return to exit')
