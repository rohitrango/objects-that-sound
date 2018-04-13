from __future__ import print_function, division
import os, cv2, json
import torch
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, Normalize, ToTensor

# Function for getting class to video map
# And video to class map
def getMappings(path1="balanced_train_segments_filtered.csv", path2="unbalanced_train_segments_filtered.csv"\
					, check_file="videoToGenre.json", videoFolder="Video"):
	# Read from files and generate mappings
	if os.path.exists(check_file):
		with open(check_file) as fi:
			vidToGenre, genreToVid = json.loads(fi.read())
		return vidToGenre, genreToVid

	# Else
	vidToGenre = dict()
	genreToVid = dict()
	for path in [path1, path2]:
		# genre to video path
		p = open(path)
		lines = p.readlines()
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			video_id = words[0]

			# Check if video is present in the folder
			if not os.path.exists(os.path.join(videoFolder, "video_" + video_id + ".mp4")):
				continue

			vidToGenre[video_id] = words[3]
			# For all genres, add the video to it
			for genre in words[3]:
				genreToVid[genre] = genreToVid.get(genre, []) + [video_id]

	# Save the file
	with open(check_file, "w+") as fi:
		fi.write(json.dumps([vidToGenre, genreToVid]))

	return vidToGenre, genreToVid


## Define custom dataset here
class GetAudioVideoDataset(Dataset):

	def __init__(self, video_path="Video/", audio_path="Audio/", transforms=None):
		self.video_path = video_path
		self.audio_path = audio_path
		self.transforms = transforms
		v2g, g2v = getMappings()

		self.vidToGenre = v2g
		self.genreToVid = g2v
		self.genreClasses = list(g2v.keys())
		self.sampleRate = 48000

		# Retrieve list of audio and video files
		for r, dirs, files in os.walk(self.video_path):
			if len(files) > 0:
				self.video_files = sorted(files)
				break

		for r, dirs, files in os.walk(self.audio_path):
			if len(files) > 0:
				self.audio_files = sorted(files)
				break

		# Print video and audio files at this point
		# print(self.video_files)
		# print(self.audio_files)

		## Calculate the number of frames and set a length appropriately

		# 40% of the total number of items are positive examples
		# 60% of the total number are negative
		# self.length --> all examples
		fps = 30
		time = 9
		tot_frames = len(self.video_files)*fps*time 
		# Frames per video
		self.fps    = fps
		self.time   = time
		self.fpv    = fps*time
		self.length = 2*tot_frames

		self._vid_transform = self._get_normalization_transform()


	def _get_normalization_transform(self):
		_vid_transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		return _vid_transform


	def __len__(self):
		# Consider all positive and negative examples
		return self.length

	def __getitem__(self, idx):
		# Given index of item, decide if its positive or negative example, and then 
		if idx >= self.length:
			print("ERROR")
			return (None, None, None)

		# Positive examples
		if idx < self.length/2:
			video_idx = int(idx/self.fpv)
			video_frame_number = idx%self.fpv
			frame_time = 500 + (video_frame_number*1000/30)

			result = [0]
			rate, samples = wav.read(os.path.join(self.audio_path, self.audio_files[video_idx]))
			# Extract relevant audio file
			time  = frame_time/1000.0
		# Negative examples
		else:
			video_idx = int((idx-self.length/2)/self.fpv)
			video_frame_number = (idx-self.length/2)%self.fpv
			frame_time = 500 + (video_frame_number*1000/30)

			result = [1]
			# Check for classes of the video and select the ones not in video
			videoID = self.video_files[video_idx].split("video_")[1].split(".mp4")[0]
			vidClasses = self.vidToGenre[videoID]
			restClasses = filter(lambda x: x not in vidClasses, self.genreClasses)
			randomClass = np.random.choice(restClasses)
			randomVideoID = np.random.choice(self.genreToVid[randomClass])
			# Read the audio now
			rate, samples = wav.read(os.path.join(self.audio_path, "audio_" + randomVideoID + ".wav"))
			time = (500 + (np.random.randint(self.fpv)*1000/30))/1000.0

		# Extract relevant frame
		#########################
		vidcap = cv2.VideoCapture(os.path.join(self.video_path, self.video_files[video_idx]))
		vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
		image = None
		success = True
		if success:
		  	success, image = vidcap.read()
		  	
		  	# Some problem with image, return some random stuff
		  	if image is None:
		  		return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2])

		  	image = cv2.resize(image, (224,224))
		  	image = image/255.0

		else:
			print("FAILURE: Breakpoint 1, video_path = {0}".format(self.video_files[video_idx]))
			return None, None, None
		##############################
		# Bring the channel to front 
		image = image.transpose(2, 0, 1)

		start = int(time*48000)-24000
		end   = int(time*48000)+24000
		samples = samples[start:end]
		frequencies, times, spectrogram = signal.spectrogram(samples, self.sampleRate, nperseg=512, noverlap=274)

		# Remove bad examples
		if spectrogram.shape != (257, 200):
			return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2])

		spectrogram = np.log(spectrogram + 1e-7)
		spec_shape = list(spectrogram.shape)
		spec_shape = tuple([1] + spec_shape)

		image = self._vid_transform(torch.Tensor(image))
		audio = torch.Tensor(spectrogram.reshape(spec_shape))
		# print(image.shape, audio.shape, result)
		return image, audio, torch.LongTensor(result)



if __name__ == "__main__":
	# a, b = getMappings()
	dataset = GetAudioVideoDataset()
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
	for (img, aud, res) in dataloader:
		print(img.shape, aud.shape, res.shape)
		print(img.max(), img.min(), aud.max(), aud.min())
	# for k in dataloader:
	# 	print(k)