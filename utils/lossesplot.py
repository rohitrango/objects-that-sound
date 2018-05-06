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
valacc = map(lambda x: float(x.split(" ")[-1]), lines)
acc = map(lambda x: float(x.split(" ")[-3]), lines)
valloss = map(lambda x: float(x.split(" ")[-7]), lines)
loss = map(lambda x: float(x.split(" ")[-9]), lines)

N = 50
acc = np.convolve(acc, np.ones((N,))/(N*1.0), mode='valid')
valacc = np.convolve(valacc, np.ones((N,))/(N*1.0), mode='valid')

loss = np.convolve(loss, np.ones((N,))/(N*1.0), mode='valid')
valloss = np.convolve(valloss, np.ones((N,))/(N*1.0), mode='valid')
# N1 = 151
# acc = medfilt(acc, N1)
# valacc = medfilt(valacc, N1)
plt.figure()
plt.plot(acc, label='accuracy')
plt.plot(valacc, label='val_accuracy')

plt.plot(loss, label='loss')
plt.plot(valloss, label='val_loss')

plt.legend(loc='lower right')
plt.show()

# print(np.max(acc), np.max(valacc), acc[-1], valacc[-1])
# med = medfilt(x, 21)
# plt.plot(med)
# plt.show()