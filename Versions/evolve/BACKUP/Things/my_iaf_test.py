import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt

import Cells


# plt.ion()

cell = Cells.iafCell()
#
# stim = h.NetStim()
# stim.number = 5
# stim.start = 10
# ncs = h.NetCon(stim, cell.soma(0.5))
# ncs.weight[0] = 1

m_vec = h.Vector()
m_vec.record(cell.soma(0.5)._ref_m)

r_vec = h.Vector()
nc = h.NetCon(cell, None)
nc.record(r_vec)

h.tstop = 50
h.run()

print m_vec.as_numpy()
print r_vec.as_numpy()
plt.plot(m_vec)
plt.show()