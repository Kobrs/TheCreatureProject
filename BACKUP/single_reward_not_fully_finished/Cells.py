from neuron import h  # not sure if we need gui
from math import sin, cos, pi
import numpy as np

class Cell(object):
    """Generic cell template."""
    def __init__(self):
        self.synlist = [] #### NEW CONSTRUCT IN THIS WORKSHEET
        self.connlist = []
        self.initialize_cell()
        self.create_sections()
        self.build_topology()
        self.build_subsets()
        self.define_geometry()
        self.define_biophysics()
        # self.create_synapses()


    def initialize_cell(self):
        """CHild can ovverride this method in order to initalize cell when created"""
        pass

    def create_sections(self):
        """Create the sections of the cell. Remember to do this
        in the form::
            h.Section(name='soma', cell=self)
        """
        raise NotImplementedError("create_sections() is not implemented.")

    def build_topology(self):
        """Connect the sections of the cell to build a tree."""
        raise NotImplementedError("build_topology() is not implemented.")

    def define_geometry(self):
        """Set the 3D geometry of the cell."""
        raise NotImplementedError("define_geometry() is not implemented.")

    def define_biophysics(self):
        """Assign the membrane properties across the cell."""
        raise NotImplementedError("define_biophysics() is not implemented.")

    def create_synapses(self):
        """Subclasses should create synapses (such as ExpSyn) at various
        segments and add them to self.synlist."""
        pass # Ignore if child does not implement.

    def build_subsets(self):
        """Build subset lists. This defines 'all', but subclasses may
        want to define others. If overridden, call super() to include 'all'."""
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)

    def connect2target(self, target, thresh=10):
        """Make a new NetCon with this cell's membrane
        potential at the soma as the source (i.e. the spike detector)
        onto the target passed in (i.e. a synapse on a cell).
        Subclasses may override with other spike detectors."""
        nc = h.NetCon(self.soma(1)._ref_v, target, sec = self.soma)
        nc.threshold = thresh
        self.connlist.append(nc)
        return nc

    def is_art(self):
        """Flag to check if we are an integrate-and-fire artificial cell."""
        return 0

    def set_position(self, x, y, z):
        """
        Set the base location in 3D and move all other
        parts of the cell relative to that location.
        """
        for sec in self.all:
            for i in range(int(h.n3d())):
                h.pt3dchange(i,
                             x - self.x + h.x3d(i),
                             y - self.y + h.y3d(i),
                             z - self.z + h.z3d(i),
                             h.diam3d(i))
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


class GenericCell(Cell):
    def create_sections(self):
        """Creates section of the cell"""
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)

    def build_topology(self):
        """Connect the section of the cell to build a tree."""
        self.dend.connect(self.soma(1))

    def define_geometry(self):
        """Define geometry of the cell. This will take parameters provided on
        object creation, so each cell can have its own defined geometry."""
        self.x = self.y = self.z = 0
        
        # Make the soma of cell a square cylinder (diameter is equal to height)
        self.soma.L = self.soma.diam = 12.6157 # microns
        self.dend.L = 200                      
        self.dend.diam = 1
        self.dend.nseg = 5

        self.shape_3D()

    def define_biophysics(self):
        """Assign membrane properties across the cell"""
        for sec in self.all:  # all is create in parent (build_subset)
            sec.Ra = 100  # axial resistance
            sec.cm = 1  # membrance capacitance in Farads/cm^2

        # Use Hodgkin-Huxley current
        self.soma.insert('hh')
        self.soma.gnabar_hh = 0.12  # sodium conductance
        self.soma.gkbar_hh = 0.036  # potassium conductance
        self.soma.gl_hh = 0.0003  # leak conductance
        self.soma.el_hh = -54.3  # reversal potentia

        # Use passive current in dendrite
        self.dend.insert('pas')
        self.dend.g_pas = 0.001
        self.dend.e_pas = -65

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

    def create_synapses(self, n=1):
        """Add an exponentially decaying synapse in the middle
        of the dendrite. Set its tau to 2ms, and append this
        synapse to the synlist of the cell."""
        for i in xrange(n):
            syn = h.ExpSyn(self.dend(0.5))
            # syn = h.ExpSyn(self.dend(round(np.random.rand(), 2)))
            syn.tau = 2
            self.synlist.append(syn) # synlist is defined in Cell




class GenericCell(Cell):
    def create_sections(self):
        """Creates section of the cell"""
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)

    def build_topology(self):
        """Connect the section of the cell to build a tree."""
        self.dend.connect(self.soma(1))

    def define_geometry(self):
        """Define geometry of the cell. This will take parameters provided on
        object creation, so each cell can have its own defined geometry."""
        
        # Make the soma of cell a square cylinder (diameter is equal to height)
        self.soma.L = self.soma.diam = 12.6157 # microns
        self.dend.L = 200                      
        self.dend.diam = 1
        self.dend.nseg = 5
        # Coordinates of this cell in 2d space. Those are public variables,
        #   not used by the class itself.

        # Initialize variable for locating cell in 1D brain space
        self.x = 0

    def define_biophysics(self):
        """Assign membrane properties across the cell"""
        for sec in self.all:  # all is create in parent (build_subset)
            sec.Ra = 100  # axial resistance
            sec.cm = 1  # membrance capacitance in Farads/cm^2

        # Use Hodgkin-Huxley current
        self.soma.insert('hh')
        self.soma.gnabar_hh = 0.12  # sodium conductance
        self.soma.gkbar_hh = 0.036  # potassium conductance
        self.soma.gl_hh = 0.0003  # leak conductance
        self.soma.el_hh = -54.3  # reversal potentia

        # Use passive current in dendrite
        self.dend.insert('pas')
        self.dend.g_pas = 0.001
        self.dend.e_pas = -65

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

    def create_synapses(self, n=1):
        """Add an exponentially decaying synapse in the middle
        of the dendrite. Set its tau to 2ms, and append this
        synapse to the synlist of the cell."""
        for i in xrange(n):
            syn = h.ExpSyn(self.dend(0.5))
            # syn = h.ExpSyn(self.dend(round(np.random.rand(), 2)))
            syn.tau = 2
            self.synlist.append(syn) # synlist is defined in Cell




class DopamineCell(GenericCell):
    def dopamine_reaction(self, dop=0.3, dopdr=0.00002):
        """This method is activated when cell is under influence of dopamine.
        It is responsble for applying any changes in activity related to it
        """

        # TEMP - only for testing dopamine mechanisms
        for syn in self.synlist:
            syn.dop = dop
            syn.dopdr = dopdr
            # syn.dop = 0.3
            # # syn.dopdr = -0.00002
            # syn.dopdr = 0.00002


    def create_synapses(self, n=1):
        """Add an exponentially decaying synapse in the middle
        of the dendrite. Set its tau to 2ms, and append this
        synapse to the synlist of the cell."""
        for i in xrange(n):
            syn = h.dop_ExpSyn(self.dend(0.5))
            # syn = h.ExpSyn(self.dend(round(np.random.rand(), 2)))
            syn.tau = 2
            self.synlist.append(syn) # synlist is defined in Cell



class STDPCell(GenericCell):
    def create_synapses(self, n=1, p=1, d=1):
        """Add an exponentially decaying synapse in the middle
        of the dendrite. Set its tau to 2ms, and append this
        synapse to the synlist of the cell."""
        for i in xrange(n):
            syn = h.ExpSynSTDP(self.dend(0.5))
            syn.tau = 2
            syn.p = p
            syn.d = d
            self.synlist.append(syn) # synlist is defined in Cell



class STDP_Dopamine_Cell(GenericCell):
    def create_synapses(self, n=1, p=1, d=1):
        """Add an exponentially decaying synapse in the middle
        of the dendrite. Set its tau to 2ms, and append this
        synapse to the synlist of the cell."""
        for i in xrange(n):
            syn = h.dop_ExpSynSTDP(self.dend(0.5))
            syn.tau = 2
            syn.p = p
            syn.d = d
            self.synlist.append(syn) # synlist is defined in Cell


    def dopamine_reaction(self, dop, dopdr):
        """This method is activated when cell is under influence of dopamine.
        It is responsble for applying any changes in activity related to it
        """

        # TEMP - only for testing dopamine mechanisms
        for syn in self.synlist:
            syn.dop = dop
            syn.dopdr = dopdr
            # syn.dop = 0.3
            # # syn.dopdr = -0.00002
            # syn.dopdr = 0.00002
