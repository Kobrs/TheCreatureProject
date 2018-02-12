import numpy as np
import binstr as b
import copy

# Genome will be encoded as a list of weights for respective cells?
# For now with static model it'll do, but ultimetely it should be rather
#   attribution of weight to target and source cell id

architecture_bin = ("111111 00000000 10000000 01001000 00000111- 00000010- 00000100-"
                "111111 00000001 10000000 01001000 00000010- 00000011- 00000100-"
                "111111 00000010 10000000 01001000 00000111- 00000101- 00000011- 00000001-"
                "111111 00000011 10000000 01001000 00000010- 00000001- 00000110-"
                "111111 00000100 10000000 01001000 00000011- 00000000- 00000001- 00000110-"
                "111111 00000101 10000000 01001000 00000111-"
                "111111 00000110 10000000 01001000 00000011-"
                "111111 00000111 10000000 01001000 00000101- 00000011- 00000110- 111111")
architecture_bin = architecture_bin.replace(" ", "")

# All of those single element tuples are meant to be replaced by iwd pair,
#   where index(i) is given as only tuple element in template below
architecture_dict_template = {
    0: {'connections': [(7,), (2,), (4,)], 
        'dend_len': 128, 'dend_pas': -55},
    1: {'connections': [(2,), (3,), (4,)],
        'dend_len': 128, 'dend_pas': -55},
    2: {'connections': [(7,), (5,), (3,), (1,)],
        'dend_len': 128, 'dend_pas': -55},
    3: {'connections': [(2,), (1,), (6,)],
        'dend_len': 128, 'dend_pas': -55}, 
    4: {'connections': [(3,), (0,), (1,), (6,)],
        'dend_len': 128, 'dend_pas': -55},
    5: {'connections': [(7,)], 'dend_len': 128, 'dend_pas': -55},
    6: {'connections': [(3,)], 'dend_len': 128, 'dend_pas': -55},
    7: {'connections': [(5,), (3,), (6,)],
        'dend_len': 128, 'dend_pas': -55}}

def interpreter(DNA):
    """Custom interpreter function for static networks
    :param DNA: string of 440 bits contating connections wd pairs"""
    architecture_dict = copy.deepcopy(architecture_dict_template)

    cell_DNA = DNA[:64]
    conn_DNA = DNA[64:]

    # First 8*8=64 bits will describe cells passive current
    for i, key in enumerate(sorted(architecture_dict.keys())):
        frame = cell_DNA[8*i:8*(i+1)]
        dend_pas = b.b_to_int(frame) - 127  # make it unsigned
        architecture_dict[key]["dend_pas"] = dend_pas



    for i in xrange(len(conn_DNA) / 20):

        wd_frame = conn_DNA[i*20:(i+1)*20]

        # print i, wd_frame

        w_type = int(wd_frame[0])
        w = wd_frame[1:16]
        w = float(b.b_to_int(w)) / 10000
        d = wd_frame[16:20]
        d = b.b_to_int(d)

        placed = False
        # Sorted below is mainly to assure consistency in conn_DNA interpretation
        #   of codons (we're forcing the order)
        for idx in sorted(architecture_dict.keys()):


            conns = architecture_dict[idx]["connections"]
            for j in xrange(len(conns)):
                if len(conns[j]) == 1:
                    architecture_dict[idx]["connections"][j] = (conns[j][0], (w_type, w), d)
                    placed = True
                    break
            if placed is True:
                break

    return architecture_dict
    


def static2dynamic_specimen(DNA, architecture_bin):
    """This function converts DNA representing static specimen onto DNA
    representing the same specimen in dynamic DNA representation. This is a
    bridge between basic implementation of GA and code written for dynamic
    representation.
    :DNA this is a binary string representing 22 connections as weight, delay
          pair (16 bits for weight and 4 bits for delay)
    """
    for i in xrange(22):
        wd = DNA[20*i:20*(i+1)]
        architecture_bin = architecture_bin.replace('-', wd, 1)

    return architecture_bin



# Following ones probably won't ever be used, but I'll leave them for now
# def encode(val_list):
#     """This is also very case specific implementation of translation from value
#     list onto grey code string.
#     Each value will be 10 000 times bigger than origanal value,then increased by
#     30 000 and converted to int"""
#     gray_string = ""
#     for val in val_list:
#         val = int(val*10000 + 30000)
#         val = val if val >= 0 else 0
#         bin_val = b.int_to_b(val, width=16)
#         gray_string += gray_val

#     return gray_val


# def create_individual():
#     """This implementation is only intended to use with basic static movement
#     circuit model"""
#     low = -0.5
#     high = 0.5
#     base_specimen = [np.random.uniform(low=low, high=high) for _ in xrange()]
#     print encode(base_specimen)

