import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt

import Cells


# plt.ion()

cell = h.IntFire4()

stim = h.NetStim()
stim.number = 5
stim.start = 10
ncs = h.NetCon(stim, cell)
ncs.weight[0] = 1

m_vec = h.Vector()
m_vec.record(cell._ref_m)

r_vec = h.Vector()
nc = h.NetCon(cell, None)
# nc = h.NetCon(stim, None)
nc.record(r_vec)

h.tstop = 50
h.run()

print m_vec.as_numpy()
print r_vec.as_numpy()
plt.plot(m_vec)
plt.show()