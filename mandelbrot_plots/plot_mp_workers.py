
# a simple script for plotting time vs workers

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Data for plotting
time = np.array([1257.044, 508.170, 263.454, 187.518, 165.476, 154.302, 147.404])
workers = np.array([1, 2, 4, 6, 8, 10, 12])

fig, ax = plt.subplots()
ax.plot(workers, time, 'r-s')

ax.set(xlabel='Workers', ylabel='Computational time (seconds)',
       title='Multiprocessing workers vs. time')

ax.grid()

fig.savefig("plot_workers_vs_time.png")
plt.show()