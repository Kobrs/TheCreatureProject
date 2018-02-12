from matplotlib import pyplot as plt
import numpy as np
import timeit
import interactive


conf = {"alpha": 0.006,
        "beta": 1.1,
        "th": 1.5,
        "momentum": 200}
im = interactive.InteractiveManager(conf)

plt.ion()

spikes = [10, 20, 25, 27, 30, 33, 36, 39, 50, 52, 55, 56, 59, 100, 120]

"""
while True:
    im.mainloop()
    plt.cla()

    momentum = int(im.conf["momentum"])

    # Analyze output data by 'emulating simulation'
    dt = 0.
    m = 0.
    t_vec = []
    m_vec = []
    if_mov = []
    updates = []
    while dt<=130:
        # I guess that this is neccessary due to rounding errrors
        dt = round(dt, 10)

        if dt in spikes:
            print "spike!"
            # m += im.conf["beta"]
            update = im.conf["beta"]
        else:
            # m -= im.conf["alpha"]
            update = -im.conf["alpha"]

        updates.append(update)
        # Update current state
        # print updates, np.mean(updates[-momentum:])
        m += np.mean(updates[-momentum:])

        if m < 0:
            m = 0


        m_vec.append(m)
        t_vec.append(dt)
        if_mov.append(True if m >= im.conf["th"] else False)

        dt += (1. / 40)
        updates = updates[-momentum:]

    plt.plot(t_vec, m_vec)
    plt.plot(t_vec, if_mov)
    plt.show()
"""
# raw_input("Press enter to exit")




"""

def detector_momentum():
    # Analyze output data by 'emulating simulation'
    momentum = im.conf["momentum"]

    dt = 0.
    m = 0.
    t_vec = []
    m_vec = []
    if_mov = []
    updates = []
    while dt<=130:
        # I guess that this is neccessary due to rounding errrors
        dt = round(dt, 10)

        if dt in spikes:
            # print "spike!"
            # m += im.conf["beta"]
            update = im.conf["beta"]
        else:
            # m -= im.conf["alpha"]
            update = -im.conf["alpha"]

        updates.append(update)
        # Update current state
        # print updates, np.mean(updates[-momentum:])
        m += np.mean(updates[-momentum:])

        if m < 0:
            m = 0


        m_vec.append(m)
        t_vec.append(dt)
        if_mov.append(True if m >= im.conf["th"] else False)

        dt += (1. / 40)
        updates = updates[-momentum:]

    # plt.plot(t_vec, m_vec)
    # plt.plot(t_vec, if_mov)
    # plt.show()


def detector_plain():
    # Analyze output data by 'emulating simulation'
    dt = 0.
    m = 0.
    t_vec = []
    m_vec = []
    if_mov = []
    while dt<=130:
        # I guess that this is neccessary due to rounding errrors
        dt = round(dt, 10)

        if dt in spikes:
            m += im.conf["beta"]
        else:
            m -= im.conf["alpha"]

        if m < 0:
            m = 0


        m_vec.append(m)
        t_vec.append(dt)
        if_mov.append(True if m >= im.conf["th"] else False)

        dt += (1. / 40)

    # plt.plot(t_vec, m_vec)
    # plt.plot(t_vec, if_mov)
    # plt.show()

# print timeit.timeit("detector()", setup="from __main__ import detector")
print timeit.timeit(detector_momentum, number=10)
print timeit.timeit(detector_plain, number=10)
"""