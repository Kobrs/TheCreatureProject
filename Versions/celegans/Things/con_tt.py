from neuron import h, gui
from matplotlib import pyplot as plt

import Cells


def create_syn(cell):
    # syn = h.dop_ExpSyn(cell.dend(0.5))
    # syn = h.ExpSyn(cell.dend(0.5))
    syn = h.ExpSynSTDP(cell.dend(0.5))
    syn.tau = 2
    cell.synlist.append(syn)


cell1 = Cells.GenericCell()
# cell1.create_synapses()
create_syn(cell1)
syn = cell1.synlist[-1]
syn.d = 0.1
syn.p = 0.1

cell2 = Cells.GenericCell()
cell2.dend.e_pas = -5


nc = cell2.connect2target(cell1.synlist[0])
nc.weight[0] = 0.002
nc.threshold = -5



stim1 = h.NetStim()
stim1.number = 1
stim1.interval = 15
stim1.start = 1

stim2 = h.NetStim()
stim2.number = 1
stim2.interval = 15
stim2.start = 10

cell1.create_synapses(1)
syn1_ = cell1.synlist[-1]
syn1_.tau = 2
nc1_ = h.NetCon(stim1, syn1_)
nc1_.weight[0] = 0.04

cell2.create_synapses(1)
syn2_ = cell2.synlist[-1]
syn2_.tau = 2
nc2_ = h.NetCon(stim2, syn2_)
nc2_.weight[0] = 0.04




rvec = h.Vector()
rvec.record(cell1.synlist[0]._ref_g)
# rvec.record(cell1.synlist[1]._ref_g)

cell_w_vec = h.Vector()
cell_w_vec.record(nc._ref_weight[0])

cell1_v_vec = h.Vector()
cell1_v_vec.record(cell1.soma(0.5)._ref_v)

cell2_v_vec = h.Vector()
cell2_v_vec.record(cell2.soma(0.5)._ref_v)


h.tstop = 70
h.run()

plt.figure(1)
plt.plot(cell1_v_vec)
plt.plot(cell2_v_vec)
plt.figure(2)
plt.plot(rvec)
plt.plot(cell_w_vec)
plt.show()



raw_input('Press return to exit')
