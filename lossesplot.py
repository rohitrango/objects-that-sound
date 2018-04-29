# coding: utf-8
from matplotlib import pyplot as plt
from scipy.signal import medfilt
import numpy as np
import sys

# bash script to copy from losses.txt to losses
# cat losses.txt  | cut -d " " -f 10


try:
	filename = sys.argv[1]
except:
	filename = "losses.txt"

with open(filename) as fi:
    lines = list(map(lambda x: x.strip("\n").replace(",", ""), fi.readlines()[:-1]))

print(len(lines))
# lines = filter(lambda x: "Epoch" not in x, lines)
# lines = map(lambda x: float(x[:-1]), lines)
# lines = np.array(lines)
valacc = list(map(lambda x: float(x.split(" ")[-1]), lines))
acc = list(map(lambda x: float(x.split(" ")[-3]), lines))

N = 35
acc = np.convolve(acc, np.ones((N,))/(N*1.0), mode='valid')
valacc = np.convolve(valacc, np.ones((N,))/(N*1.0), mode='valid')
# N1 = 151
# acc = medfilt(acc, N1)
# valacc = medfilt(valacc, N1)


plt.figure()
plt.plot(acc, label='accuracy')
plt.plot(valacc, label='val_accuracy')
plt.legend(loc='upper left')
plt.show()

# med = medfilt(x, 21)
# plt.plot(med)
# plt.show()