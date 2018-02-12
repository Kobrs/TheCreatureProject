import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

plt.ion()

# spikes = [10, 20, 25, 27, 30, 33, 36, 39, 50, 52, 55, 56, 59, 100, 120]
spikes = [10, 20, 25, 27, 30, 33, 36, 39]

def model(x, t, spikes):
    # if t<10:
    #     C = 1
    # else:
    #     C = 0

    print t
    if t in spikes:
        C = 1
        print "asdfzxcv"
    else:
        C = 0.01

    dxdt = -0.1 * x + C

    print dxdt

    return dxdt


# initial condition
y0 = 0

# time points
t = np.linspace(0,40, num=40)

# # solve ODE
# y = odeint(model,y0,t, args=(spikes,), tcrit=spikes)
y = odeint(model,y0,t, args=(spikes,), tcrit=spikes)

# plot results
plt.plot(t, y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()

raw_input("Enter to close")