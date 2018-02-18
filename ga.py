import numpy as np
import binstr as b
from neuron import h, gui
from matplotlib import pyplot as plt
import Cells
try:
    from py2cytoscape.data.cyrest_client import CyRestClient
    from py2cytoscape.data.util_network import NetworkUtil as util
    import networkx as nx
except ImportError:
    print "Graphing libs not available!"

try:
    from deap import base, creator, tools
    from scoop import futures
except ImportError:
    print "Deap not available - evolutionary algorithms no available"
import copy
import time  # TEMP


class GAinterpreter(object):
    def __init__(self, frame_len=6):
        self.frame_len = frame_len
        self.w_len = 12
        self.d_len = 4
        # NOTE
        self.id_len = 8
        self.coord_len = 8
        # Following frame code is all ones because it is quite unlikely for
        #   actual variables to reach such mariginal values, so those will be
        #   pretty unique, especially for longer frames.
        self.frame_code = '00' + '1'*(frame_len-2)
        self.Cells = {}

    def _find_frames(self):
        f_codes_i = []
        i = 0
        while i <= len(self.DNA) - (self.frame_len-1):
            codon = self.DNA[i:i+self.frame_len]
            if codon == self.frame_code:
                # print "frame opened at index:", i
                f_codes_i.append(i)
            i+=1

        return f_codes_i

    def _decode_frame(self, frames_id):
        prev_id = None
        for fid in frames_id:
            if prev_id is None:
                prev_id = fid
            elif fid - prev_id >= 52+self.frame_len:
                frame = self.DNA[prev_id + self.frame_len:fid]
                # print "interpreting frame:", frame
                self._interpret_frame(frame)
                prev_id = fid



    def _decode_iwd_triple(self, iwd_frame):
        i0 = 0
        i1 = self.id_len
        c_id = iwd_frame[i0:i1]
        c_id = b.b_to_int(c_id)

        i0 = i1
        i1 += 1
        w_type = int(iwd_frame[i0:i1])

        i0 = i1
        i1 += self.w_len
        w = iwd_frame[i0:i1]
        # Weights can only be positive
        w = float(b.b_to_int(w)) / 10000.

        i0 = i1
        i1 += self.d_len
        d = iwd_frame[i0:i1]
        d = b.b_to_int(d)

        return c_id, (w_type, w), d


    def _interpret_frame(self, frame):
        """First 8 bits will be the cell id, next 8 bits will be dendrite len,
        further 8 bits will be dendrite passive current(int8). Everything after
        that will be target id, weight and delay triple - id have 8bits,
        weight have 16 divided to two parts: 1 bit for weight sign and 15 bits
        for value multiplied by 10 000 bit and delay 4. This triplet will
        have 28 bit"""

        i0 = 0
        i1 = self.id_len
        cell_id = frame[i0:i1]
        cell_id = b.b_to_int(cell_id)

        i0 = i1
        i1 += self.coord_len
        x = frame[i0:i1]
        x = b.b_to_int(x)

        # i0 = i1
        # i1 += self.coord_len
        # y = frame[i0:i1]
        # y = b.b_to_int(y)

        i0 = i1
        i1 += 8  # dend len
        dend_len = frame[i0:i1]
        dend_len = b.b_to_int(dend_len) + 1  # it cannot be 0

        i0 = i1
        i1 += 8
        dend_pas = frame[i0:i1]
        dend_pas = b.b_to_int(dend_pas) - 127

        # Scan for (index, weight, delay) triples
        iwd_len = self.id_len + 1 + self.w_len + self.d_len  # 1 is w_type
        iwd_triples = []
        i = i1
        while i < len(frame)-(iwd_len-1):
            iwd_frame = frame[i: i+iwd_len]
            iwd_triples.append(self._decode_iwd_triple(iwd_frame))

            i+=iwd_len

        cell_dict = {"x": x, "dend_len": dend_len, "dend_pas": dend_pas,
                     "connections": iwd_triples}
        # Overwrite existing settings or create new
        self.Cells[cell_id] = cell_dict


    def decode(self, DNA):
        # First 26 bits are encoding of global dopamine parameters, so we treat
        #   them separately.
        dop = b.b_to_int(DNA[0:10]) / 100.
        dopdr = b.b_to_int(DNA[10:26]) / 655360.
        self.DNA = DNA[26:]
        frames_id = self._find_frames()
        self._decode_frame(frames_id)

        return self.Cells, dop, dopdr




# TODO: try if instead of creating object for each specimen and delete it after
#       measuring fitness, I'd make a one object and add them mehtods to reset
#       variables to initilal state and use it to measure perofrmance of all
#       sepcimens given their DNA.
class Specimen(GAinterpreter):
    def __init__(self, DNA, cell, num_of_regions, cells_per_pain_stimuli=3,
                 interpreter=None,frame_len=6, graph=False, logging=True):
        """Object for simulating evolved architecture.
        :param architecture: dictionary contaitng connections, dend_len, dend_pas
        :param cell: cell object, implementation based on Cell base class
        :param num_of_regions: Specify number of regions to be used. There will
            be eaual size. This should include undefined regions if present.
        :param logging: True if log infomrations

        posisble usages:
        * through fitness mehtod, which does basically everythting
        * manual, which allows plotting and graphing:
            spec.construct_network()
            spec.show_architecture()
            spec.set_recording_vectors()
            spec.apply_stimuli({0: [5, 15, 50, 55, 70, 72, 74],
                               1: [5, 15, 50, 55, 70, 72, 74]})
            spec.setup_output_cells([5, 6])
            # spec.graph_connections()
            spec.run(100)
            print spec.get_out_spikes()
            spec.plot_cells_activity(mode=2)"""
        if interpreter is None:
            super(Specimen, self).__init__(frame_len)
            self.architecture, dop, dopdr = self.decode(DNA)
        else:
            self.architecture, dop, dopdr = interpreter(DNA)

        self.cell = cell
        self.cells_per_pain_stimuli = cells_per_pain_stimuli
        self.graph = graph
        self.dop = dop
        self.dopdr = dopdr
        self.num_of_regions = num_of_regions
        self.logging = logging

        self.cells_dict = {}
        self.conn_list = []  # contains 3 elemnt tuple: source, target, weight
        self.gate_cells = {}
        self.record_vectors_dict = {}
        self.stim_conns = []
        self.vecStims = []
        self.stim_trains = []
        self.out_spike_dict = {}
        self.out_spike_virt_conns = []
        self.constructed = False

        self.output_data = []

        if graph is True:
            self._setup_graph()

        self.construct_network()


    def _log(self, val):
        if self.logging:
            print val



    def construct_network(self):
        if self.constructed:
            raise Exception("Network already constructed!")
        # Define cellls - connections are possible only between defined cells
        for cell_id, cell_prop in self.architecture.iteritems():
            cell = self.cell()
            cell.dend.L = cell_prop["dend_len"]
            cell.dend.e_pas = cell_prop["dend_pas"]

            cell.x = cell_prop["x"]

            self.cells_dict[cell_id] = cell


        # Define connections
        for src_cell_id, src_cell in self.cells_dict.iteritems():
            # each connection in tgt_conns is (id, weight, delay) triple
            tgt_conns = self.architecture[src_cell_id]["connections"]
            for i, iwd in enumerate(tgt_conns):
                w_type, w = iwd[1]

                try:
                    tgt_cell = self.cells_dict[iwd[0]]
                    tgt_cell.create_synapses(1)
                    syn = tgt_cell.synlist[-1]
                    if w_type == 0:  # synapse is inhibitory
                        syn.e = -70
                    else:
                        syn.e = 0.0  # synapse is excitatory

                    nc = src_cell.connect2target(syn)
                    nc.weight[0] = w
                    nc.delay = iwd[2]

                    # We have to do this, otherwise NEURON will lost the connection
                    #   we've just made.
                    self.conn_list.append((src_cell_id, iwd[0], nc))

                    if self.graph is True:
                        # self.g.add_edge(iwd[0], src_cell_id)
                        self.g.add_edge(src_cell_id, iwd[0])
                        self._log("adding edge:", src_cell_id, iwd[0])
                        self._log(self.g.edges())
                except KeyError:
                    # It tries to connect to cell that doesn't exist
                    pass

        self.constructed = True


    def add_stim_gate_cell(self, cell_id, targets):
        """
        This function creates cell used as gateway between sensory cell and
        direct current stimulation.
        :param cell_id: id of newly created gete cell. Note that they are stored
                        separately from main cells
        :param targets: list of targets ids.
        """
        cell = Cells.GenericCell()
        for target in targets:
            self.cells_dict[target].create_synapses(1)
            syn = self.cells_dict[target].synlist[-1]
            # We assume that tey are excitatory
            syn.e = 0.
            nc = cell.connect2target(syn)
            nc.weight[0] = 0.04  # weight is constant
            nc.delay = 0  # no delay
            nc.threshold = -20  # gate cells have weirdly low maximum potential

            self.conn_list.append(nc)

            self.gate_cells[cell_id] = cell


    def stimulate_gate_cell(self, cell_id, amp, dur):
        """
        Stimulate cells added in add_stim_gate_cell with iclamp with provided
        current. Current should be in range 0.15 to 0.4
        """
        cell = self.gate_cells[cell_id]
        clamp = h.IClamp(cell.dend(0.1))
        clamp.dur = dur
        clamp.amp = amp

        self._log("stimulating with current = %f"%amp)

        # So that it'll stay around only until it's deleted in clear_stim
        self.stim_conns.append(clamp)


    def add_noise_to_cells(self, cells, mean=0.01, stdev=0.3, dur=1e9,
                           gate_cell=False, one_step=False):
        """
        Add gaussian noise current to given cells
        :param cells: list of cells on which noise will be applied
        :param mean: mean of noise gaussian distribution
        :param stdev: standard deviation of noise guassian distribution
        :param dur: duration of noise
        :param gate_cells: indicate whether we gave index to normal cells or to
                           gate cells
        :param one_step: if true make noise disappear after single simulation
        """
        for cell_id in cells:
            cell = self.cells_dict[cell_id] if not gate_cell else self.gate_cells[cell_id]
            noise = h.InGauss(cell.dend(0.1))
            noise.mean = mean
            noise.stdev = stdev
            noise.dur = dur

            # If one step is true, then add to stim_conns, which will be reset
            #   after each step
            if one_step:
                # Add it somewhere so it stays until clear_stim is called
                self.stim_conns.append(noise)
            else:
                self.conn_list.append(noise)



    def apply_stimuli(self, stim_dict, weight=0.5):
        """This function applies stimuli based on stim_dict parameters.
        :param stim_dict: key is target cell id and value is spike times list"""
        for tgt_cell_id, times in stim_dict.iteritems():
            # try:
            if tgt_cell_id in self.cells_dict.keys():
                tgt_cell = self.cells_dict[tgt_cell_id]
                # DEBUG: The following line causes slowdown!!!
                #        This isn't due to creation time though, rather to
                #        the effect it has on simulation itslef.
                syn = tgt_cell.create_synapses(1, return_syns=True)[0]

                stim_train = h.Vector(times)

                vplay = h.VecStim()  # NOTE: this requires compiled vecstim.mod
                vplay.play(stim_train)
                nc = h.NetCon(vplay, syn)
                nc.weight[0] = weight

                # Now note, that if I'm going to be re running simulation
                #   and reinitalize it each time, the time will reset and
                #   each time network will be stimulated by old stimuli.
                self.stim_conns.append((nc, syn))
                self.vecStims.append(vplay)
                self.stim_trains.append(stim_train)
            # except KeyError:
            #     pass


    def setup_output_cells(self, cells_id):
        """This funciton setups recording of spikes on the output cells
        "param cells_id: list of cell ids concidered output cells"""
        for cell_id in cells_id:
            try:
                cell = self.cells_dict[cell_id]
                r_vec = h.Vector()
                nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
                nc.record(r_vec)

                self.out_spike_dict[cell_id] = r_vec
                self.out_spike_virt_conns.append(nc)
            except KeyError:
                pass


    def get_out_spikes(self):
        """This function returns output spike times as a dictionary, where
        key is cell id and value is spike times array"""
        out_dict = {}
        for cell_id, spikes in self.out_spike_dict.iteritems():
            out_dict[cell_id] = np.round(spikes.to_python(), decimals=6)
        return out_dict


    def release_dopamine(self, region, dop=None, dopdr=None):
        """This function will release dopamine into different brain regions.
        One of them will never under influence of dopamine and doesn't have
        specified function.
        This function is cmpatible with brain cantainign five regions, where
        the fifth one is concidered to be free region, on which no control
        mechanisms are aplied.
        :param region: id of brain region where dopamine should be released."""


        dop = dop if dop is not None else self.dop
        dopdr = dopdr if dopdr is not None else self.dopdr

        region_width = 255 / self.num_of_regions  # coords are in reange 0-255

        for cell in self.cells_dict.values():
            # We divide brain area to five segments region_width units each
            if int(cell.x / region_width) == region:
                # Find all connections from this cell
                cell.dopamine_reaction(dop, dopdr)



    def feel_the_pain(sefl, source):
        """This function applies stimuli to the cells defined as pain cells.
        :param source: index of source, it will be used to determine location
                       of pain cells"""
        first_cell = 128 + source*self.cells_per_pain_stimuli
        stim_dict = {}

        for i in xrange(self.cells_per_pain_stimuli):
            stim_dict[first_cell+i] = [1]  # stimulate cell as soon as possible

        self.apply_stimuli(stim_dict)



    def show_architecture(self):
        shape_window = h.PlotShape()
        shape_window.exec_menu("Show Diam")

    def set_recording_vectors(self, record_dend=False, record_t=True):
        for key, cell in self.cells_dict.iteritems():
            soma_v_vec = h.Vector()   # Membrane potential vector at soma
            soma_v_vec.record(cell.soma(0.5)._ref_v)

            if record_dend:
                dend_v_vec = h.Vector()   # Membrane potential vector at dendrite
                dend_v_vec.record(cell.dend(0.5)._ref_v)
            else:
                dend_v_vec = []  # just create something to add to dictionary

            if record_t:
                t_vec = h.Vector()        # Time stamp vector
                t_vec.record(h._ref_t)
            else:
                t_vec = []

            self.record_vectors_dict[key] = [t_vec, soma_v_vec, dend_v_vec]

    def plot_cells_activity(self, mode=2, set_scale=True, show=True, fig=None):
        dend_plot = None
        if fig is None: fig = plt.figure(16, figsize=(8,4))
        legend_plots = []
        legend_labels = []

        for cell_id, rvec in self.record_vectors_dict.iteritems():
            t_vec, soma_v_vec, dend_v_vec = rvec
            if mode == 0:
                soma_plot = plt.plot(t_vec, soma_v_vec, color='black')
                dend_plot = plt.plot(t_vec, dend_v_vec, color='red')
            elif mode == 1:
                soma_plot = plt.plot(t_vec, soma_v_vec)
                dend_plot = plt.plot(t_vec, dend_v_vec)
            elif mode == 2:
                soma_plot = plt.plot(t_vec, soma_v_vec)
                legend_plots.append(soma_plot[0])  # soma_plot is a tuple
                try:
                    label = "(%s)"%self.architecture[cell_id]['label']
                except KeyError:
                    label = ""
                legend_labels.append("Cell" + str(cell_id) + label)
            else:
                self._log("[ERROR] Wrong plotting mode!")
                raise SystemExit

        if set_scale:
            ax = fig.axes[0]
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(range(int(start), int(end), 5))


        if dend_plot is not None:
            plt.legend(soma_plot + dend_plot, ['soma', 'dend(0.5)'])
        else:
            plt.legend(legend_plots, legend_labels)

        plt.xlabel('time (ms)')
        plt.ylabel('mV')

        if show: plt.show()


    def run(self, time=25):
        h.tstop = time
        h.run()


    def sim_init(self):
        """Take a look into advance function docstring."""
        h.stdinit()
        # h.init()


    def clear_sim(self):
        # Clear stuff that's not useful anymore
        # self.stim_conns = []
        # self.vecStims = []
        # self.stim_trains = []
        del self.stim_conns[:]
        del self.vecStims[:]
        del self.stim_trains[:]


    def advance(self, time):
        """
        Function to be used if calling NEURON simulation multiple times.
        NOTE: To get any data out of this simulation make sure that sim_init
        method is called at least once befeore advance calls, but after
        applying stimuli and setting record vectors.
        """
        h.continuerun(time)


    def _setup_graph(self):
        self.cy = CyRestClient()
        self.cy.session.delete()

        self.CY_STYLE = self.cy.style.create("Sample3")
        defaults = {
            "NODE_BORDER_PAINT": "black",
            "NODE_LABEL_COLOR": "#33FF33",
            "NODE_LABEL_FONT_SIZE": "18",
            "NODE_SIZE": "30",
            "EDGE_SOURCE_ARROW_SHAPE": "arrow",
            "EDGE_SOURCE_ARROW_UNSELECTED_PAINT": "#9999FF",
            "EDGE_STROKE_COLOR": "#CCCCCC"
        }
        self.CY_STYLE.update_defaults(defaults)

        self.g = nx.DiGraph()

    def graph_connections(self):
        # Auromatically initialize graph if it's not initalized already
        if self.graph is False:
            self._setup_graph()

        g_cy = self.cy.network.create_from_networkx(self.g)
        self.cy.style.apply(style=self.CY_STYLE, network=g_cy)
        self.cy.layout.apply(network=g_cy)
        self.cy.layout.fit(network=g_cy)

        view_id_list = g_cy.get_views()
        view = g_cy.get_view(view_id_list[0], format="view")

        idmap = util.name2suid(g_cy)
        shape_dict = {}
        color_dict = {}
        border_dict = {}
        for n_id in idmap.values():
            shape_dict[n_id] = "ellipse"
            color_dict[n_id] = "white"
            border_dict[n_id] = "#FF56E9"


        view.update_node_views(visual_property="NODE_SHAPE", values=shape_dict)
        view.update_node_views(visual_property="NODE_FILL_COLOR", values=color_dict)
        view.update_node_views(visual_property="NODE_BORDER_PAINT", values=border_dict)


    def fitness_score(self, stimuli, output_cells, reflex=20, interia=5, sim_time=None):
        """This function scores current specmien based on output from
        simulation. This function is NOT intended to be universal. It will
        marginally differ between applications, therefore programmer is
        encourage to use public, high level methods from current class.
        It is also encouraged to create local functions inside of this method
        and not add functions used only for this block to the class"""

        # stimuli = {0: [5, 15, 50, 55, 70, 72, 74],
        #            1: [5, 15, 50, 55, 70, 72, 74]}

        # Determine simulation time
        if sim_time is None:
            try:
                sim_time = max([max(t) for t in stimuli.values()]) + 10
            except ValueError:
                sim_time = 120


        # This is intended to be standalone function, so we perfrom simulation
        #   first
        self.construct_network()
        self.show_architecture()
        # self.set_recording_vectors()
        self.apply_stimuli(stimuli)
        self.setup_output_cells(output_cells)
        self.run(sim_time)
        # self.plot_cells_activity(mode=2)

        # Dictionary of out_cell_id: spike_times_array
        spikes = self.get_out_spikes()


        """Current decoding of signal: each spike on either right or left
        neuron produces forward movement of given direction for k ms(I use
        implementation more common for wheeled systems, where each side of
        vechicle is controled separately)."""

        current_stimuli = None
        score = 0
        ambiguous_movement = False
        timeline = []
        mov_record = []
        timers_record = []
        # index 0 is for left forward, 1 is left backward, 2 for right forward
        #   3 for right backward
        mov = [False, False, False, False]
        mov_timers = [0, 0, 0, 0]  # timers to measure movement elapsed time
        reflex_timers = [0, 0]  # track response time to stimuli
        dt = round(h.dt, 4)
        t = 0


        # Helper rewards for having crucial cells in architecture:
        # for input cells
        scored = dict([(c_id, False) for c_id in output_cells+stimuli.keys()])
        for in_cell in stimuli.keys():
            if in_cell in self.architecture.keys():
                scored[in_cell] = True
        # for output cells
        for out_cell in output_cells:
            for val in self.architecture.values():
                if out_cell in [iwd[0] for iwd in val['connections']]:
                    scored[out_cell] = True
        score = sum(scored.values()) * 50


        while t<=sim_time:
            # Get rid of rounding error on further decimal places
            t = round(t, 5)

            # Note that we assume that spikes dict have the same number of
            #   cells as we have movements possible
            for i, sp_list in enumerate(spikes.values()):
                for sp_time in sp_list:
                    if t == round(sp_time, 4):
                        # self._log("movement")
                        # Start movement and reset timer
                        mov[i] = True
                        mov_timers[i] = interia


            for i in xrange(len(mov)):
                if round(mov_timers[i], 5) == 0:
                    mov[i] = False
                elif mov_timers[i] > 0.:
                    mov_timers[i] -= dt



            # Check if we have any stimuli at current time
            for i, key in enumerate(sorted(stimuli.keys())):
                # This thing is to ensure that we have reflex_timers set in
                #   right order, which is crucial for deciding if specimen
                #   should be rewarded or not.
                times = stimuli[key]
                for sp in times:
                    if sp == t:
                        reflex_timers[i] = reflex



            for i, timer in enumerate(reflex_timers):
                if timer > 0:
                    # Reward it for moving backward
                    if mov[1] and mov[3] is True and i==0:
                        score +=0.05

                    # Reward it for moving forward
                    elif mov[0] and mov[2] is True and i==1:
                        score +=0.05

                    # Reward it for moving at all
                    if (mov[1] and mov[3]) ^ (mov[0] and mov[2]):
                        score += 0.005

                    # Reward it for doing anything connected to stimuli
                    if mov[0] or mov[1] or mov[2] or mov[3]:
                        score += 0.0001
                else:
                    # Reset timer
                    reflex_timers[i] = 0

                # Decrease timer for next iteration of while loop
                reflex_timers[i] -= dt


            timeline.append(t)
            mov_record.append(copy.copy(mov))
            timers_record.append(copy.copy(mov_timers))

            t += dt

        self.output_data = [timeline, mov_record, timers_record]


        return score + 0.1  # because it cannot be 0



def scoop_creator_init():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    return creator


class GA(object):
    def __init__(self, mutate_rate=0.05, toursize=3, pop_size=25, CX_max_len_diff=700,
                 MUTPB=0.2, FSL=4, DMPB=0.2, TRPB=0.8, creator=None, scoop=False):
        """Class which actually performs evolution
        :param mutate_rate: rate of mutation of bit in single specimen
        :param toursize: size of tournament used for selecting mates
        :param pop_size: size of the population
        :param CX_max_len_diff: maximal difference of lenght between crossing specimens
        :param MUTPB: probability of mutating given specimen
        :param FSL: the length for the flanking sequences
        :param DMPB: duplicate and move probability: probability of duplicating
                     and moving further when it is 'moving'
        :param TRPB: probability of performing transposition on specimen
        :param creator: initialized in global scope creator object - needed for scoop
        :param scoop: Use scoop distributed computation if true.
        """
        self.mutate_rate = mutate_rate
        self.toursize = toursize
        self.pop_size = pop_size
        self.MUTPB = MUTPB
        self.CX_max_len_diff = CX_max_len_diff
        self.FSL = FSL
        self.DMPB = DMPB
        self.TRPB = TRPB
        self.creator = creator
        self.scoop = scoop
        self.toolbox = base.Toolbox()

    def _mutate(self, individual, indpb):
        # NOTE: individual(specified in function registered as 'individual')
        #       have to be a binstring
        new_individual = ""
        for bit in individual:
            if np.random.rand() < indpb:
                new_individual += '0' if bit == '1' else '1'
            else:
                new_individual += str(bit)

        return new_individual,

    def _transpose(self, individual, FSL=4, DMPB=0.2):
        """Another genetic operator. It chooses some transposon and exchanges
        it between two parents creating offspring. It uses aproach from
        "Transposition: A Biologically Inspired Mechanism to Use with Genetic
        Algorithms", with a difference that we use it alongside with two point
        crossover, therefore we don't swap transposons, but instead as
        described in Biological Transposition section we randomly choose
        whether transposon should move, copy and move and whether when it moves
        should it duplicate, insert and continue or just stop.
        :param individual: individual on which this will be aplied, it can be
                           either bit string or array of bools.
        :param FSL: the length for the flanking sequences
        :param DMPB: duplicate and move probability: probability of duplicating
                     and moving further when it is 'moving' """

        CL = len(individual)
        T = np.random.randint(0, CL)
        # Frame sequence
        FS = individual[T-FSL:T]
        # Frame ending
        FE = None

        # Indexes of location of the last bit for all flags in individual
        FLoc = []
        # Find all flags, incuding ending of current one
        # for i in xrange(T, CL-(FSL-1), 1):
        for i in xrange(CL-(FSL-1)):
            frame = individual[i:i+FSL]
            if frame == FS or frame == FS[::-1]:
                if FE is None and i > T:
                    FE = i+FSL
                else:
                    FLoc.append(i+FSL)

        if FE is not None:
            # Decide whether copy and move or just move
            if np.random.rand()<0.5:
                # Just move
                transposon = individual[T:FE]
                individual = individual[:T] + individual[FE:]
            else:
                # Copyt and move
                transposon = individual[T:FE]

            try:
                while np.random.rand()<DMPB:
                    tgt = np.random.choice(FLoc)
                    individual = individual[:tgt] + transposon + individual[tgt:]
            except ValueError:
                pass

        return individual



    def create_functions(self, generate_individual, evaluate_func):
        """This method creates all functions neccesary to run deap GA.
        :param generate_individual: a function returning array of specimen DNA
        :param evaluate_func: function returning score of specimen NOTE: it
                              MUST return single element tuple"""

        toolbox = self.toolbox

        if self.creator is not None:
            creator = self.creator  # overwrite creator import in this scope

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox.register("individual",
                         lambda container: container(generate_individual()),
                         creator.Individual)
        if self.scoop:
            toolbox.register("map", futures.map)
        else:
            toolbox.register("map", map)

        toolbox.register("population", tools.initRepeat, list,
                         toolbox.individual)
        toolbox.register("evaluate", evaluate_func)
        # This crossover function is safe to use with variable length DNA
        toolbox.register("mate", tools.cxTwoPoint)
        # NOTE: we're assuming that individual is list of bools
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutate_rate)
        # toolbox.register("mutate", self._mutate, indpb=self.mutate_rate)
        toolbox.register("transpose", self._transpose, FSL=self.FSL, DMPB=self.DMPB)
        # toolbox.register("select", tools.selTournament, tournsize=self.toursize)
        # toolbox.register("select", tools.selDoubleTournament, fitness_size=self.toursize, parsimony_size=1.4, fitness_first=False)
        toolbox.register("select", tools.selRoulette)

    def evolve(self):
        toolbox = self.toolbox  # ugly, but we'll use toolbox here a lot
        # First create population

        pop = toolbox.population(n=self.pop_size)

        # Evaluate entire population
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        fits = [ind.fitness.values[0] for ind in pop]

        g = 0
        # Run as long as user don't stop this program
        while True:
            g += 1
            print "--Generation %i --"%g

            offspring = toolbox.select(pop, len(pop))
            # Clone selected offspring
            offspring = list(toolbox.map(toolbox.clone, offspring))

            # TODO: Modify this code so mating two similar size specimen is more likely than largely varying ones

            # Mate
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability depending on len difference
                cross_prob = abs(len(child1) - len(child2)) / float(self.CX_max_len_diff)
                if np.random.rand() < cross_prob:
                    toolbox.mate(child1, child2)
                    # Clear previous fitnesses
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutate
            for mutatnt in offspring:
                if np.random.rand() < self.MUTPB:
                    toolbox.mutate(mutatnt)
                    del mutatnt.fitness.values
            # Transpose
            for transposed in offspring:
                if np.random.rand() < self.TRPB:
                    toolbox.transpose(transposed)
                    del transposed.fitness.values

            # Evaluated newly created offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, [g]*len(invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]


            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print "Lengths:",
            for i in pop:
                print "%d,"%len(i),
            print ""

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
