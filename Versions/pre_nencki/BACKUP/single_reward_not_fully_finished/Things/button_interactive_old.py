import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import thread


class ControlPanel(object):
    def __init__(self, param_dict):
        self.param_dict = param_dict

        # Use some ridiculusly numbered figure to avoid overwriting user's figs
        plt.figure(128)

        bool_count = 0
        float_count = 0
        # Count each type appearance
        for key, val in param_dict.iteritems():
            # For binary values
            if type(val) == bool:
                bool_count += 1
            elif type(val) == float:
                float_count += 1

        try:
            bool_increase = 1. / bool_count
        except ZeroDivisionError:
            bool_increase = 0

        try: 
            float_increase = 1. / float_count
        except ZeroDivisionError:
            float_increase = 0
        # # Always take smaller height increacement
        # h_increase = bool_increase if bool_increase < float_increase else float_increase

        # Keep count on how many we've already drawn
        i_b = 0
        i_f = 0
        bool_labels = []
        check_states = []
        # Create figure and fill it with gui elemeents
        for key, val in param_dict.iteritems():
            # For binary values
            if type(val) == bool:
                # b_ax = plt.axes([0.05, 0.05+(bool_increase*i_b), 0.1, 0.05])
                bool_labels.append(key)
                check_states.append(val)

            elif val is float:
                float_count += 1

        if bool_increase != 0:
            b_ax = plt.axes([0.05, 0.05, 0.5, 0.95])
            print bool_labels, check_states
            b = CheckButtons(b_ax, bool_labels, check_states)
            b.on_clicked(self.bool_change)

        # This should be put into new thread
        plt.show()


    def bool_change(self, event):
        self.param_dict[event] = not self.param_dict[event]




c = ControlPanel({"asdf": True})
raw_input("enter to close")