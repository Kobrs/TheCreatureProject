n_cells = 10
sensor_cells = [1, 2]
# Note that tracks cells are reversed, so it works with off active stimulation
trackL_cell = 3
trackR_cell = 4
gate_cells = [-1, -2]
dend_pas = -60.

def generate_architecture_all2all(n, w_min=0, w_max=0.1, d_min=0, d_max=16):
    """
    Generates all to all network architecture with random weights in given range
    """
    architecture = {}
    for cell in xrange(n):
        # Generate all2all connections
        # x = 0
        # x = np.random.randint(0, 256)
        x = 255 if cell in [trackL_cell, trackR_cell] else 0
        conns = []
        for i in xrange(n):
            w = np.random.uniform(low=w_min, high=w_max)
            w_type = np.random.randint(low=0, high=2)
            d = np.random.randint(low=d_min, high=d_max)
            conns.append((i, (w_type, w), d))

        architecture[cell] = {'connections': conns, 'x': x, 'dend_len':100,
                              'dend_pas': dend_pas}
    return architecture


def generate_architecture_prebuilt(n=None, w_min=0, w_max=0.1, d_min=0, d_max=16):
    d = 1
    specimen_architecture = {
        sensor_cells[0]: {'connections': [(8, (1, 0.04), d), (9, (0, 0.3), d), (sensor_cells[1], (0, 0.0), d)],
            'x': 0, 'dend_len': 100, 'dend_pas': -65, 'label': 'AWCL'},
        sensor_cells[1]: {'connections': [(10, (0, 0.3), d), (11, (1, 0.04), d), (sensor_cells[0], (0, 0.0), d)],
            'x': 0, 'dend_len': 100, 'dend_pas': -65, 'label': 'AWCR'},
        8: {'connections': [(trackL_cell, (1, 0.01), d)],
            'x': 0, 'dend_len': 100, 'dend_pas': -65, 'label': 'AIBL'},
        9: {'connections': [(12, (1, 0.0005), d)],
            'x': 0, 'dend_len': 100, 'dend_pas': 2000, 'label': 'AIYL'},  # this huge pas current will lead to self spiking cell
        10: {'connections': [(12, (1, 0.0005), d)],
            'x': 0, 'dend_len': 100, 'dend_pas': 2000, 'label': 'AIYR'},
        11: {'connections': [(trackR_cell, (1, 0.01), d)],
            'x': 0, 'dend_len': 100, 'dend_pas': -65, 'label': 'AIBR'},
        12: {'connections': [(trackL_cell, (1, 0.04), d), (trackR_cell, (1, 0.04), d)],
            'x': 0, 'dend_len': 100, 'dend_pas': -65, 'label': 'X'},
        trackL_cell: {'connections': [],
            'x': 0, 'dend_len': 100, 'dend_pas': -65, 'label': 'ML'},
        trackR_cell: {'connections': [],
            'x': 0, 'dend_len': 100, 'dend_pas': -65, 'label': 'MR'}
    }

    return specimen_architecture


def generate_architecture_braitenberg():
	# Connection contains: 'target cell', indicator of conn type(i,e), weight and delay values
	specimen_architecture = {sensor_cells[0]: {'connections': [(trackL_cell, (1, 0.015), 5)],
		 'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
		sensor_cells[1]: {'connections': [(trackR_cell, (1, 0.03), 5)],
		 'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
		trackL_cell: {'connections': [],
		 'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
		trackR_cell: {'connections': [],
		 'x': 0, 'dend_len': 100, 'dend_pas': dend_pas}}

	return specimen_architecture
