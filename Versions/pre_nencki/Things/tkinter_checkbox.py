from Tkinter import *

master = Tk()

# var = IntVar(value=1)
var = BooleanVar(value=False)


c = Checkbutton(master, text="Expand", variable=var, onvalue=True, offvalue=False, padx=10, pady=5)
c.pack()

mainloop()