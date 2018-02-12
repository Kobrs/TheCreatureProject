import numpy as np
import matplotlib.pyplot as plt

mu = 2.5
sigma = 0.6


s = np.random.normal(mu, sigma, 50)
print s
print np.array(s, dtype=int)

x = np.arange(0, 5, 0.01)
# y = np.random.normal()
plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (x - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

plt.show()

