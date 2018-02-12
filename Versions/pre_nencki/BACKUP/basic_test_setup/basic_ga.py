import numpy as np
import binstr

# Genome will be encoded as a list of weights for respective cells?
# For now with static model it'll do, but ultimetely it should be rather
#   attribution of weight to target and source cell id

architecture = ("111111 00000000 10000000 01001000 00000111- 00000010- 00000100-"
                "111111 00000001 10000000 01001000 00000010- 00000011- 00000100-"
                "111111 00000010 10000000 01001000 00000111- 00000101- 00000011- 00000001-"
                "111111 00000011 10000000 01001000 00000010- 00000001- 00000110-"
                "111111 00000100 10000000 01001000 00000011- 00000000- 00000001- 00000110-"
                "111111 00000101 10000000 01001000 00000111-"
                "111111 00000110 10000000 01001000 00000011-"
                "111111 00000111 10000000 01001000 00000101- 00000011- 00000110- 111111")
architecture = architecture.replace(" ", "")

def encode(val_list):
    """This is also very case specific implementation of translation from value
    list onto grey code string.
    Each value will be 10 000 times bigger than origanal value,then increased by
    30 000 and converted to int"""
    gray_string = ""
    for val in val_list:
        val = int(val*10000 + 30000)
        val = val if val >= 0 else 0
        bin_val = binstr.int_to_b(val, width=16)
        gray_string += gray_val

    return gray_val


def create_individual():
    """This implementation is only intended to use with basic static movement
    circuit model"""
    low = -0.5
    high = 0.5
    base_specimen = [np.random.uniform(low=low, high=high) for _ in xrange()]
    print encode(base_specimen)


def static2dynamic_specimen(DNA, architecture):
    """This function converts DNA representing static specimen onto DNA
    representing the same specimen in dynamic DNA representation. This is a
    bridge between basic implementation of GA and code written for dynamic
    representation.
    :DNA this is a binary string representing 22 connections as weight, delay
          pair (16 bits for weight and 4 bits for delay)
    """
    for i in xrange(22):
        wd = DNA[20*i:20*(i+1)]
        architecture = architecture.replace('-', wd, 1)

    return architecture

