import Tkinter as tk
import thread


class ControlPanel(object):
    def __init__(self, conf):
        self.conf = conf




        thread.start_new_thread(self.loop, ())



    def loop(self):
        self.master = tk.Tk()
        self.master.title("Control Panel")

        # Dictionary to store state variables - it can be used to change button state
        self.state_vars = {}
        # Create figure and fill it with gui elemeents
        rows = [0, 0]
        for key, val in self.conf.iteritems():
            # For binary values

            if type(val) == bool:
                self.state_vars[key] = tk.BooleanVar(value=val)
                c = tk.Checkbutton(self.master, text=key, command=lambda key=key: self.bool_change(key),
                                   variable=self.state_vars[key], onvalue=True,
                                   offvalue=False, padx=10, pady=5)
                c.grid(column=0, row=rows[0])
                rows[0] += 1

            elif type(val) == float:
                self.state_vars[key] = tk.DoubleVar(value=val)
                # Add entry and button for this entry
                l = tk.Label(self.master, text=key)
                e = tk.Entry(self.master, textvariable=self.state_vars[key])
                b = tk.Button(self.master, text="ok", command=lambda key=key, value=self.state_vars[key]: self.float_change(key, value))
                l.grid(column=1, row=rows[1])
                e.grid(column=2, row=rows[1])
                b.grid(column=3, row=rows[1])
                rows[1] += 1

        tk.mainloop()


    def bool_change(self, key):
        self.conf[key] = not self.conf[key]

    def float_change(self, key, value):
        value = value.get()
        print key, value
        self.conf[key] = value

    def set_state(self, key, state):
        self.state_vars[key].set(state)
        self.conf[key] = state




# Usage:

# c = ControlPanel({"asdf": True, "zxcv":False})
# # raw_input("enter to close")
# while True:
#     print c.conf