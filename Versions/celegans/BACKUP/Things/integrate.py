import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


u_rest = -70
R = 0.1
I = 1.5

# function that returns dy/dt
def model(y,t):
    # k = 0.3
    # dydt = -k * y
    dydt = -(y - u_rest) + R*I
    return dydt

# initial condition
y0 = u_rest

# time points
t = np.linspace(0,20)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
