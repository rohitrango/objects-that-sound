# coding: utf-8
from matplotlib import pyplot as plt
from scipy.signal import medfilt

# bash script to copy from losses.txt to losses
# cat losses.txt  | cut -d " " -f 10
with open("losses") as fi:
    lines = fi.readlines()

lines = lines[:-1]
lines = filter(lambda x: "Epoch" not in x, lines)
lines = map(lambda x: float(x[:-1]), lines)
import numpy as np
lines = np.array(lines)

x = np.convolve(lines, np.ones((20,))/20.0, mode='valid')
plt.plot(x)
plt.show()

med = medfilt(x, 21)
plt.plot(med)
plt.show()