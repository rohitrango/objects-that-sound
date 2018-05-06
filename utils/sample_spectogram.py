# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
from scipy import signal

def get_spect(audio_file_name):
	audio_file_path = audio_file_name
	sample_rate, samples = wav.read(audio_file_path)
	# It should be nperseg = 480 and overlap = 240 ? But I get 241 x 199
	# Below values give 257 x 200
	time = 5.0
	start = int(time*48000)-192000
	end   = int(time*48000)+192000

	frequencies, times, spectrogram = signal.spectrogram(samples[start:end], sample_rate, nperseg=512, noverlap=274)
	return frequencies,times,spectrogram

audio_file_names = ["audio_xxNroISqkt4.wav", "audio_zj2G-KVw4N4.wav", "audio_yOJtRE-617I.wav", "audio_yfaxqwNHe7w.wav", "audio_yoSL2qcfbcc.wav", "audio_zcdr8KnM_hM.wav", "audio_yrtihXfrYx0.wav", "audio_zCqpwFfsvXQ.wav", "audio_woVby8SBWDI.wav", "audio_x23YOCp2w0U.wav", "audio_wZZx0zTb0Xw.wav", "audio_wst-3U_wU3g.wav", "audio_xsbANTuPp4k.wav", "audio_y-G_mOM5LbQ.wav", "audio_yURIdR7A1oM.wav", "audio_zXKT561l8u8.wav", "audio_vx3FnzyOZv8.wav", "audio_x3nrjdqBbOg.wav", "audio_yZWx2IW34Wg.wav", "audio_yp4YBTyX7gk.wav"]
labels = ["Accordion","Accordion","Bagpipes","Bagpipes","Chant","Chant","Child singing","Child singing","Choir","Choir","Clarinet","Clarinet","Cowbell","Cowbell","Dental drill, dentist's drill","Dental drill, dentist's drill","Electronic organ","Electronic organ","Church bell","Church bell"]

plt.figure()

for i,audio_file_name in enumerate(audio_file_names):
	# if i%2:
	# 	pass
	frequencies,times,spectrogram = get_spect(audio_file_name)
	print(spectrogram.shape)

	plt.subplot(5, 4, 1 + i)
	plt.pcolormesh(times, frequencies, spectrogram)
	plt.imshow(np.log(spectrogram))
	plt.xlabel(labels[i])
	print(spectrogram.shape)

plt.show()
