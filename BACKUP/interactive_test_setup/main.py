import numpy as np
from neuron import h, gui
from matplotlib import pyplot as plt
import simrun
import Cells
import interactive as i

# plt.switch_backend('Qt5Agg')


# TODO:
# So now, when we have basic framework to work with I can define desired
#   architecture and start working GA implementation.


cell1 = Cells.MechanosensoryCell()
cell1.dend.L = 100
cell2 = Cells.MechanosensoryCell()
cell2.dend.L = 300
cell2.dend.e_pas = -65


shape_window = h.PlotShape()
shape_window.exec_menu("Show Diam")

# Create a spike generator
stim = h.NetStim() 


syn_ = h.ExpSyn(cell1.dend(0.5))


# Create synapse and attach it to the end of dendrite
syn_ = h.ExpSyn(cell1.dend(0.5))
syn_.tau = 2

stim.number = 10
stim.interval = 20
stim.start = 10

ncstim = h.NetCon(stim, syn_)
ncstim.delay = 1
ncstim.weight[0] = 0.04

# syn = h.ExpSyn(cell2.dend(0.5))
# nc = h.NetCon(cell1.soma(0.5)._ref_v, syn, sec=cell1.soma)
# nc.weight[0] = 0.2
# nc.delay = 2
syn = cell2.synlist[0]
nc = cell1.connect2target(syn)
nc.weight[0] = 2
nc.delay = 2


soma_v_vec1, dend_v_vec1, t_vec1 = simrun.set_recording_vectors(cell1)
soma_v_vec2, dend_v_vec2, t_vec2 = simrun.set_recording_vectors(cell2)

fig = plt.figure(figsize=(8,4))


conf = {"w": 0.04,
        "cell2 e_pass": -65}
im = i.InteractiveManager(conf)

plt.ion()
while True:
    im.mainloop()
    plt.cla()

    nc.weight[0] = im.conf["w"]
    cell2.dend.e_pas = im.conf["cell2 e_pass"]

    simrun.simulate(150)

    # simrun.show_output(soma_v_vec1, dend_v_vec1, t_vec1, fig, mode=1)
    simrun.show_output(soma_v_vec2, dend_v_vec2, t_vec2, fig, mode=1)

    print "showing figure"
    plt.show()

# figManager = plt.get_current_fig_manager()
# figManager.window.move(200, 200)
# figManager.window.showMaximized()
# figManager.window.setFocus()