import numpy as np
from neuron import h, gui
from math import sin, cos, pi
from matplotlib import pyplot as plt
from neuronpy.graphics import spikeplot
from neuronpy.util import spiketrain
import simrun


class BallAndStick(object):
    """Two-section cell: A soma with active channels and
    a dendrite with passive properties."""
    def __init__(self):
        self.x = self.y = self.z = 0
        self.create_sections()
        self.build_topology()
        self.build_subsets()
        self.define_geometry()
        self.define_biophysics()

    def create_sections(self):
        """Create the sections of the cell."""
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)

    def build_topology(self):
        """Connect the sections of the cell to build a tree."""
        self.dend.connect(self.soma(1))

    def define_geometry(self):
        """Set the 3D geometry of the cell."""
        self.soma.L = self.soma.diam = 12.6157 # microns
        self.dend.L = 200                      # microns
        self.dend.diam = 1                     # microns
        self.dend.nseg = 5
        self.shape_3D()    #### Was h.define_shape(), now we do it.

    def define_biophysics(self):
        """Assign the membrane properties across the cell."""
        for sec in self.all: # 'all' exists in parent object.
            sec.Ra = 100    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
        # Insert active Hodgkin-Huxley current in the soma
        self.soma.insert('hh')
        self.soma.gnabar_hh = 0.12  # Sodium conductance in S/cm2
        self.soma.gkbar_hh = 0.036  # Potassium conductance in S/cm2
        self.soma.gl_hh = 0.0003    # Leak conductance in S/cm2
        self.soma.el_hh = -54.3     # Reversal potential in mV
        # Insert passive current in the dendrite
        self.dend.insert('pas')
        self.dend.g_pas = 0.001  # Passive conductance in S/cm2
        self.dend.e_pas = -65    # Leak reversal potential mV

    def build_subsets(self):
        """Build subset lists. For now we define 'all'."""
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)

    def shape_3D(self):
        """
        Set the default shape of the cell in 3D coordinates.
        Set soma(0) to the origin (0,0,0) and the dend extrending 
        along the x-axis.
        """
        len1 = self.soma.L
        # Destroy location info in soma
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0, 0, 0, self.soma.diam, sec=self.soma)
        h.pt3dadd(len1, 0, 0, self.soma.diam, sec=self.soma)
        len2 = self.dend.L
        h.pt3dclear(sec=self.dend)
        h.pt3dadd(len1, 0, 0, self.dend.diam, sec=self.dend)
        h.pt3dadd(len1 + len2, 0, 0, self.dend.diam, sec=self.dend)

    def set_position(self, x, y, z):
        """
        Set the base location in 3D and move all other parts of the cell
        relative to that location.
        """
        for sec in self.all:
            # this loop changes constest for all neuron functions that
            #   depend on section, so need to specify sec=
            for i in xrange(int(h.n3d(sec=sec))):
                h.pt3dchange(i, 
                             x - self.x + h.x3d(i, sec=sec),
                             y - self.y + h.y3d(i, sec=sec),
                             z - self.z + h.z3d(i, sec=sec),
                             h.diam3d(i, sec=sec), sec=sec)
        self.x, self.y, self.z = x, y, z

    def rotateZ(self, theta):
        """Rotate the cell about the Z axis"""
        for sec in self.all:
            for i in xrange(int(h.n3d(sec=sec))):
                x = h.x3d(i, sec=sec)
                y = h.y3d(i, sec=sec)
                c = cos(theta)
                s = sin(theta)
                xprime = x*c - y*s
                yprime = x*s + y*c
                h.pt3dchange(i, xprime, yprime, h.z3d(i, sec=sec),
                             h.diam3d(i, sec=sec), sec=sec)


cells = []
N = 10
r = 50  # Radius of cell locations from (0, 0, 0) in microns
for i in xrange(N):
    cell = BallAndStick()
    # When cells are created, the soma location is at (0,0,0) and
    # the dendrite extends along the x-axis.
    # First, at the origin, rotate about Z
    cell.rotateZ(i*2*pi/N)
    # Then reposition
    x_loc = cos(i*2*pi/N)*r
    y_loc = sin(i*2*pi/N)*r
    cell.set_position(x_loc, y_loc, 0)
    cells.append(cell)

shape_window = h.PlotShape()
shape_window.exec_menu("Show Diam")

raw_input("")


# stim = h.NetStim()  # Make a spike generator

# # Attach it to a synapse in the middle of the dendrite of the first cell int
# #   the network. (Named 'syn_' to avoid being overwritten wiht the syn var
# #   used later.)
# # syn_ = h.ExpSyn(cells[0].dend(0.5), name='syn_')
# syn_ = h.ExpSyn(cells[0].dend(0.5))

# stim.number = 1
# stim.start = 9
# ncstim = h.NetCon(stim, syn_)
# ncstim.delay = 1
# ncstim.weight[0] = 0.04  # NetCon weight is a vector
# syn_.tau = 2


# src = cells[0]
# tgt = cells[1]
# syn = h.ExpSyn(tgt.dend(0.5))
# nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
# nc.weight[0] = 0.04
# nc.delay = 5

# soma_v1, dend_v1, t1 = simrun.set_recording_vectors(src)
# soma_v2, dend_v2, t2 = simrun.set_recording_vectors(tgt)

# simrun.simulate(100)

# fig = plt.figure()
# simrun.show_output(soma_v1, dend_v1, t1, fig, mode=1)
# simrun.show_output(soma_v2, dend_v2, t2, fig, mode=1)


# # Connect the cells
# nclist = []
# syns = []
# for i in xrange(N):
#     src = cells[i]
#     tgt = cells[(i+1)%N]
#     syn = h.ExpSyn(tgt.dend(0.5))
#     syns.append(syn)
#     nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
#     nc.weight[0] = 0.04
#     nc.delay = 5
#     nclist.append(nc)


# # Current recording variables
# syn_i_vec = h.Vector()
# syn_i_vec.record(syn_._ref_i)

# soma_v_vec_list = []
# dend_v_vec_list = []
# for cell in cells:
#     soma_v_vec, dend_v_vec, t_vec = simrun.set_recording_vectors(cell)
#     soma_v_vec_list.append(soma_v_vec)
#     dend_v_vec_list.append(dend_v_vec)

# t_vec_nc = h.Vector()
# id_vec = h.Vector()
# for i in range(len(nclist)):
#     nclist[i].record(t_vec_nc, id_vec, i)

# simrun.simulate(100)


# fig = plt.figure(figsize=(8,4)) # Default figsize is (8,6)
# for i in xrange(len(cells)):
#     simrun.show_output(soma_v_vec_list[i], dend_v_vec_list[i], t_vec, fig,
#                        mode=1)



# fig = plt.figure(figsize=(8,4))
# ax2 = fig.add_subplot(1,1,1)
# syn_plot = ax2.plot(t_vec, syn_i_vec, color='blue')
# ax2.legend(syn_plot, ['synaptic current'])
# ax2.set_ylabel(h.units('ExpSyn.i'))
# ax2.set_xlabel('time (ms)')


# spikes = spiketrain.netconvecs_to_listoflists(t_vec_nc, id_vec)
# print spikes
# sp = spikeplot.SpikePlot(savefig=True)
# sp.plot_spikes(spikes)

plt.show()