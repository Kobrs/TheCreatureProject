"""
Thiis code is based and complementary to main.py. It is not bounded to 
creature body simulation, but instead it allows to run simulation of given
architecture with desired parameters and create raster plot with net activity.
Designed to confirm and evaluate model behavior.
"""

import numpy as np
from neuron import h
from matplotlib import pyplot as plt

import Cells
import ga
import architecture

plt.ion()

sim_time = 500 #  [ms]
# Lamp = 0.3
# Ramp = 0.3
motor_max_spikes = 6
engine_force = 0.15
engine_time_window = 50
noise_mean = 0.01
noise_stdev = 0.1
plotting_activity = True

n_cells = architecture.n_cells
sensor_cells = architecture.sensor_cells
trackL_cell = architecture.trackL_cell
trackR_cell = architecture.trackR_cell
gate_cells = architecture.gate_cells

specimen_architecture = architecture.generate_architecture_prebuilt()


# Allow mapping plot order
information_flow = [1, 2, 8, 9, 10, 11, 12, 3, 4]
offset_dict = dict([(v, i) for (i, v) in enumerate(reversed(information_flow))])


def val_map(x, in_min, in_max, out_min, out_max):
    x = float(x)  # make sure that we get float operations
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def raster_plot(spikes):
	# Create raster plot. Spikes is dictionary of {key: times}
	data = []
	offsets = []
	colors = []
	labels = []
	for neuron, times in spikes.iteritems():
		data.append(times)
		# offsets.append(neuron)
		offsets.append(offset_dict[neuron])
		colors.append([np.random.rand() for _ in xrange(3)])
		labels.append(specimen_architecture[neuron]['label'])

	rplot = plt.eventplot(data, lineoffsets=offsets, color=colors,
						  linelengths=[0.4], linewidths=[2])
	plt.legend(rplot, labels)

	plt.xlabel('spike times')
	plt.ylabel('neuron id')
	# plt.yticks(range(min(spikes.keys()), max(spikes.keys())+1))
	label_ticks = labels
	x = offsets
	plt.yticks(x, label_ticks)


def force_plot(spikes):
	# Now plot motor force
	forces = []
	for cell in sensor_cells:
		forces.append([])
		spike_train = spikes[cell]
		for i in xrange(sim_time / engine_time_window):
			current_spikes = []
			for sp in spike_train:
				if i*engine_time_window < sp < (i+1)*engine_time_window:
					current_spikes.append(sp)

			force = val_map(len(current_spikes), 0, motor_max_spikes, 0, engine_force)
			forces[-1].append(force)


	x = np.arange(sim_time/engine_time_window+1) * engine_time_window
	Lforce = np.array(forces[0]+[forces[0][-1]])
	Rforce = np.array(forces[1]+[forces[1][-1]])
	plt.plot(x, Lforce, label='Left track force')
	plt.plot(x, Rforce, label='Right track force')
	plt.legend(loc='upper right')


def prepare_fig(rows, cols, ratio=None):
	if ratio is None:
		f, plots = plt.subplots(rows, cols)
	else:
		f, plots = plt.subplots(rows, cols,  gridspec_kw={'height_ratios':ratio})

	return plots


def full_spike_plot(spikes, plots, plot_id, title=""):
	plot_id -= 1  # convert to 1 index so its consitant with matplotlib

	plt.axes(plots[0][plot_id])
	plt.title(title)
	# plt.subplot(2, num_plots, plot_id, gridspec_kw={'height_ratios':[3,1]})
	raster_plot(spikes)
	
	plt.axes(plots[1][plot_id])
	# plt.subplot(2, num_plots, plot_id+num_plots, gridspec_kw={'height_ratios':[3,1]})
	force_plot(spikes)

	# raise SystemExit


def plot_activity(record_vectors_dict, plots, plot_id, title=""):
	plot_id -= 1  # convert to 1 index so its consitant with matplotlib

	for cell_id, rvec in record_vectors_dict.iteritems():
		t_vec, soma_v_vec, dend_v_vec = rvec
		i = len(offset_dict.keys()) - offset_dict[cell_id] - 1
		ax = plots[i, plot_id]
		if i == 0: ax.set_title(title)
		ax.plot(t_vec, soma_v_vec)
		ax.legend([specimen_architecture[cell_id]['label']], loc='upper left')


def get_activity(Lamp, Ramp, plot=False):
	interpreter = lambda _: (specimen_architecture, 0, 0)
	net = ga.Specimen("", Cells.STDP_Dopamine_Cell, interpreter=interpreter,
	                  num_of_regions=1)

	net.add_stim_gate_cell(gate_cells[0], [sensor_cells[0]])
	net.add_stim_gate_cell(gate_cells[1], [sensor_cells[1]])

	net.stimulate_gate_cell(gate_cells[0], Lamp, sim_time)
	net.stimulate_gate_cell(gate_cells[1], Ramp, sim_time)

	# Add gaussian noise to gate cells
	net.add_noise_to_cells(gate_cells, mean=noise_mean, stdev=noise_stdev,
						  gate_cell=True)

	# Setup spike recording for all cells
	net.setup_output_cells(net.cells_dict.keys())

	net.set_recording_vectors()


	# Run the simulation
	net.run(time=sim_time)


	spikes = net.get_out_spikes()

	return spikes, net.record_vectors_dict



plots1 = prepare_fig(2, 3, ratio=[3, 1])
if plotting_activity: plots2 = prepare_fig(len(specimen_architecture.keys()), 3)


# Situation I
title = "First case - both sensory neurons evenly stimulated with high current"
spikes, record_vectors_dict = get_activity(Lamp=0.3, Ramp=0.3)
full_spike_plot(spikes, plots1, 1, title=title)
if plotting_activity:
	plot_activity(record_vectors_dict, plots2, 1, title=title)

# Situation II
title = "Second case - both sensory neurons evenly stimulated with low current"
spikes, record_vectors_dict = get_activity(Lamp=0.055, Ramp=0.055)
full_spike_plot(spikes, plots1, 2, title=title)
if plotting_activity:
	plot_activity(record_vectors_dict, plots2, 2, title=title)


# Situation III
title = "Third case - left neuron stimulated with high current, right with low"
spikes, record_vectors_dict = get_activity(Lamp=0.3, Ramp=0.04)
full_spike_plot(spikes, plots1, 3, title=title)
if plotting_activity:
	plot_activity(record_vectors_dict, plots2, 3, title=title)




plt.show()

raw_input("Press return to exit")
