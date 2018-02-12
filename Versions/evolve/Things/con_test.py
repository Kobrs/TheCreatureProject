import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt

import Cells


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
# cell1.dend.e_pas = 1000

cell2 = Cells.GenericCell()
# cell2.dend.e_pas = 500

cell3 = Cells.GenericCell()


# Add stimuli to both cells
ssyn1 = create_syn(cell1)
# ssyn1.e = -70  # inhibitory
sns1 = h.NetStim()
sns1.interval = 10
sns1.number = 5
sns1.start = 1
snc1 = h.NetCon(sns1, ssyn1)
snc1.weight[0] = 0.04

ssyn1_ = create_syn(cell3)
snc1_ = h.NetCon(sns1, ssyn1_)
snc1_.weight[0] = 0.04


# ssyn2 = create_syn(cell2)
# sns2 = h.NetStim()
# sns2.interval = 20
# sns2.number = 3
# sns2.start = 5
# snc2 = h.NetCon(sns2, ssyn2)
# snc2.weight[0] = 0.04


# # Addditional stimuli to prevent regularity
# ssyn3 = create_syn(cell1)
# sns3 = h.NetStim()
# sns3.interval = 50
# sns3.number = 3
# sns3.start = 200
# snc3 = h.NetCon(sns3, ssyn3)
# snc3.weight[0] = 0.04



# Connect both cells with STDP synapse
cell2.create_synapses(1)  # cell1 is presynaptic
nc = cell1.connect2target(cell2.synlist[-1])  # cell2 is postsynaptic
nc.weight[0] = 0.0005

# # Connect both cells with STDP synapse
cell2.create_synapses(1)  # cell1 is presynaptic
nc2 = cell3.connect2target(cell2.synlist[-1])  # cell2 is postsynaptic
nc2.weight[0] = 0.001



# Setup recording vectors
cell1_v_vec = h.Vector()
cell1_v_vec.record(cell1.soma(0.5)._ref_v)

cell2_v_vec = h.Vector()
cell2_v_vec.record(cell2.soma(0.5)._ref_v)

# cell1_w_vec = h.Vector()
# cell1_w_vec.record(cell1.synlist[-1]._ref_g)

tvec = h.Vector()
tvec.record(h._ref_t)



h.tstop = 70
h.run()

plt.figure(1)
plt.plot(tvec, cell1_v_vec)
plt.plot(tvec, cell2_v_vec)
# plt.figure(2)
# plt.plot(tvec, cell1_w_vec)
plt.show()



raw_input('Press return to exit')
