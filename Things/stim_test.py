import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt

import Cells


plt.ion()


cell = Cells.GenericCell()
syn = h.ExpSyn(cell.dend(0.5))
syn.tau = 2

# clamp = h.SEClamp(cell.soma(0.5))
# clamp.vc = 10
# clamp.i = 0.1
# clamp.dur1 = 100
# clamp.i = 0

# cell.dend.e_pas = 0
clamp = h.IClamp(cell.dend(0.1))
# clamp.del = 0
clamp.dur = 200
clamp.amp = 0.04

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
