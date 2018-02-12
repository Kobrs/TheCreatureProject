import Tkinter as tk
import thread


class ControlPanel(object):
    def __init__(self, conf):
        self.conf = conf




        thread.start_new_thread(self.loop, ())



    def loop(self):
        self.master = tk.Tk()

        # Dictionary to store state variables - it can be used to change button state
        self.state_vars = {}
        # Create figure and fill it with gui elemeents
        for key, val in self.conf.iteritems():
            # For binary values

            self.state_vars[key] = tk.BooleanVar(value=val)


            if type(val) == bool:
                c = tk.Checkbutton(self.master, text=key, command=lambda key=key: self.bool_change(key),
                                   variable=self.state_vars[key], onvalue=True,
                                   offvalue=False, padx=10, pady=5)
                c.pack(side=tk.BOTTOM)

            elif type(val) == float:
                print "Floats aren't implemented yet, ignoring!"

        tk.mainloop()


    def bool_change(self, key):
        self.conf[key] = not self.conf[key]


    def set_state(self, key, state):
        self.state_vars[key].set(state)
        self.conf[key] = state




# Usage:

# c = ControlPanel({"asdf": True, "zxcv":False})
# # raw_input("enter to close")
# while True:
#     print c.conf