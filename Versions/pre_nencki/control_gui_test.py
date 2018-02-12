import ControlGUI
import time

conf = {"dopamine-r0": False,
        "dopamine-r1": False,
        "dopamine-r2": False,
        "asdf": 0.1
        }
i = ControlGUI.ControlPanel(conf)

# raw_input('')

while True:
    print i.conf
    time.sleep(0.01)