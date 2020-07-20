import matplotlib.pyplot as plt
import time
import random
from collections import deque
import numpy as np

# simulates input from serial port


def random_gen():
    while True:
        val = random.randint(1, 10)
        yield val
        time.sleep(0.1)


f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(np.linspace(1, 100, 100), np.linspace(1, 100, 100))

a1 = deque([0] * 100)
f2 = plt.figure()
ax = f2.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
d = random_gen()

# line, = ax.plot([], [])
plt.ion()
plt.ylim([0, 10])
plt.show()

for i in range(0, 20):
    # line.set_data([i, i + 5], [i, i + 5])
    ax.scatter([i, i + 5], [i, i + 5], s=2)
    plt.draw()
    i += 1
    time.sleep(0.1)
    plt.pause(0.0001)
