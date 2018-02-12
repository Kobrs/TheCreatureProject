import numpy as np
# from neuron import h, gui
from neuron import h
from matplotlib import pyplot as plt
import time
import timeit 

import simrun
import Cells
import interactive
import ga
import simple_creature



plt.ion()


# def create_syn(cell):
#     # syn = h.dop_ExpSyn(cell.dend(0.5))
#     # syn = h.ExpSyn(cell.dend(0.5))
#     syn = h.ExpSynSTDP(cell.dend(0.5))
#     syn.tau = 2
#     cell.synlist.append(syn)


# cell1 = Cells.GenericCell()
# # cell1.create_synapses()
# create_syn(cell1)
# syn = cell1.synlist[-1]
# syn.d = 0.1
# syn.p = 0.1

# cell2 = Cells.GenericCell()
# # cell2.dend.e_pas = -5


# nc = cell2.connect2target(cell1.synlist[0])
# nc.weight[0] = 0.002
# nc.threshold = -5



# stim1 = h.NetStim()
# stim1.number = 1
# stim1.interval = 15
# stim1.start = 1

# stim2 = h.NetStim()
# stim2.number = 1
# stim2.interval = 15
# stim2.start = 10

# cell1.create_synapses(1)
# syn1_ = cell1.synlist[-1]
# syn1_.tau = 2
# nc1_ = h.NetCon(stim1, syn1_)
# nc1_.weight[0] = 0.04

# cell2.create_synapses(1)
# syn2_ = cell2.synlist[-1]
# syn2_.tau = 2
# nc2_ = h.NetCon(stim2, syn2_)
# nc2_.weight[0] = 0.04




# rvec = h.Vector()
# rvec.record(cell1.synlist[0]._ref_g)
# # rvec.record(cell1.synlist[1]._ref_g)

# cell_w_vec = h.Vector()
# cell_w_vec.record(nc._ref_weight[0])

# cell1_v_vec = h.Vector()
# cell1_v_vec.record(cell1.soma(0.5)._ref_v)

# cell2_v_vec = h.Vector()
# cell2_v_vec.record(cell2.soma(0.5)._ref_v)


# h.tstop = 70
# h.run()

# plt.figure(1)
# plt.plot(cell1_v_vec)
# plt.plot(cell2_v_vec)
# plt.figure(2)
# plt.plot(rvec)
# plt.plot(cell_w_vec)
# plt.show()



# raw_input('Press return to exit')





sim_dt = 1./40

DNA_str = "101101010101111011111000010011110000000001000100001000000001001000000000001010000100110100011001111" 
net = ga.Specimen(DNA_str, Cells.DopamineCell, num_of_regions=3)
print net.architecture
print net.cells_dict
net.set_recording_vectors()
net.apply_stimuli({0: [sim_dt*400*i for i in xrange(100)]})

# Note that this have to be called BEFORE apply_stimuli and set_recording_vectors
# net.sim_init()  # TEMP

def run():
    for i in xrange(10000):
        net.advance(time=h.t+sim_dt)
        # net.run(time=h.t+sim_dt)

def run_():
    net.run(time=h.t+10000*sim_dt)


def run_and_init():
    # Bare minimum initialization
    h.init()

    net.advance(time=h.t+10000*sim_dt)
    # for i in xrange(10000):
    #     net.advance(time=h.t+sim_dt)


# This doesn't behave in the same way as...
h.running_ = 1
h.stdinit()
h.continuerun(h.t+100)

h.running_ = 1
h.stdinit()
h.continuerun(h.t+100)


# ...this, even though it should be identical.
net.run(time=h.t+100)
# net.run(time=h.t+100)



# net.sim_init()
# h.init()
# net.run(time=h.t+100)
# net.run(time=h.t+100)

# t1 = timeit.timeit(run_, number=2)

# raw_input("Enter to contunue!")


# t2 = timeit.timeit(run_and_init, number=2)

# # run_and_init()

# print "initialization time:", timeit.timeit(h.init, number=1)  # -> around 3.60012054443e-05s
# print "initialization time:", timeit.timeit(h.stdinit, number=1)  # -> around 2.98023223877e-05s

net.plot_cells_activity(set_scale=False)

# print "We get following timings:", t1, t2, "t1 is", t1/t2, "times bigger than t2."


raw_input('Press return to exit')

































# TODO: TAKE SOME SCREENSHOTS OF THIS THING IN ACTION, BECAUSE I WON'T BE ABLE TO DO IT REMOTELY!
#       THIS MAY BE USEFUL FOR EXPLORY POSTER!


























