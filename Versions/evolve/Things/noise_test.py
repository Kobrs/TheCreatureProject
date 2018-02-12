import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt

import Cells


plt.ion()


cell = Cells.GenericCell()
syn = h.ExpSyn(cell.dend(0.5))
syn.tau = 2

noise = h.InGauss(cell.soma(0.5))
noise.mean = 0.01
noise.stdev = 0.4
noise.dur = 10000


r_vec = h.Vector()
nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
nc.record(r_vec)
nc.threshold = -20


cell_v_vec = h.Vector()
cell_v_vec.record(cell.soma(0.5)._ref_v)

tvec = h.Vector()
tvec.record(h._ref_t)


h.tstop = 50
h.run()


print r_vec.as_numpy()

plt.plot(tvec, cell_v_vec)
plt.show()

raw_input('Press return to exit')
