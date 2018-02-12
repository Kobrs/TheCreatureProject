import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = plt.plot(t, s, lw=2)


class Index(object):
    ind = 0
    check_states = [True]

    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def checked(self, event):
        print self.check_states[0]


callback = Index()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
axcheck = plt.axes([0.5, 0.05, 0.1, 0.075])

bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)

bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

check_box = CheckButtons(axcheck, ['check'], callback.check_states)
check_box.on_clicked(callback.checked)

plt.show()