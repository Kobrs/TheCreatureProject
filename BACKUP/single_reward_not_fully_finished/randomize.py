import numpy as np


def random_connect(architecture_dict, p=0.5, w_low=0, w_high=0.2, d_low=0, d_high=10):
    """This function generates randomly connecte network of neurons connecting
    two neurons with given probability.
    :param cells: list of cells to be connected.
    :param architecture_dict: dictionary with defined architecture except
                              connections field, which will be replaced.
    :param p: probability of connecting two cells.
    :returns: architecture dictionary filled with connections.
    """


    for cellA in architecture_dict.keys():
        connections = []
        for cellB in architecture_dict.keys():
            if np.random.rand() < p:
                i = cellB
                w_type = np.random.randint(0, 2)
                w = np.random.uniform(low=w_low, high=w_high)
                d = np.random.uniform(low=d_low, high=d_high)

                connections.append((i, (w_type, w), d))

        architecture_dict[cellA]['connections'] = connections

    return architecture_dict



def create_architecture(cells):
    """Creates architecture where all cells have hardcoded parameters, but
    connections are not specified.
    :param cells: list of cells to be used."""

    architecture = {}
    for cell in cells:
        architecture[cell] = {'dend_len': np.random.uniform(low=0, high=120), 'dend_pas': -55}

    return architecture