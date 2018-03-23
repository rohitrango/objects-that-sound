# coding: utf-8
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import json
import youtube_dl
import subprocess
import time
import threading
import os

class myThread (threading.Thread):
	def __init__(self, lin, count):
		threading.Thread.__init__(self)
		self.lin = lin
		self.count = count

	def run(self):
		print ("Starting " + self.name)
		download_vid(self.lin,self.count)
		print ("Exiting " + self.name)

vid2class = dict()

def download_vid(lin, count):

	# Extract the words consisting of video_id, start_time, end_time, list of video_tags
	words =  [word.replace("\n","") for word in lin.split(",")]
	words = words[0:3] + [words[3:]]
	video_id = words[0]

	vid2class[video_id] = words[3]

	ydl_opts = {
		'start_time': int(float(words[1])),
		'end_time': int(float(words[2])),
		'format': 'best[height<=360]',
		'outtmpl' : r"{}_{}.%(ext)s".format("full",video_id)
	}

	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		ydl.download(['https://www.youtube.com/watch?v=' + video_id])
		
		info = ydl.extract_info(str('https://www.youtube.com/watch?v=' + video_id), download=False) # Extract Info without download
		ext = info.get('ext',None) # Extract the extension of the downloaded video
		
		subprocess.call(["ffmpeg","-ss",str(int(float(words[1]))),"-i","full_" + video_id +"."+ext,"-t","00:00:10","-vcodec","copy","-acodec","copy", "video_" + video_id + "."+ext])

		# Video to Audio Conversion 
		# -i is for input file
		# -ab is bit rate
		# -ac is no of channels
		# -ar is sample rate
		# -vn is no video
		audio_file_path = "audio_" + video_id + ".wav"
		command = ["ffmpeg", "-i", "video_"+video_id+"."+ext,"-ab","160k", "-ac","1","-ar","44100","-vn",audio_file_path]
		subprocess.call(command)
	print("Im Done")


# Lines for every video
with open("balanced_train_segments_filtered.csv") as f:
	lines = f.readlines()

# Load all tags for checking download
with open('tags.cls') as file:
	tags = map(lambda x: x[:-1], file.readlines())

print(tags)

threads = []
i = 0

for i in range(len(lines)):

	if len(threads) == 3:
		for t in threads:
			t.join()
			print "Joined thread"
		threads = []
		print "Joined Threads"
		os.system("rm full*")
		os.system("rm *.webm")
		os.system("mv *.mp4 Video/")
		os.system("mv *.wav Audio/")

		# subprocess.call(command)
		# command = ["rm", "*.webm"]
		# subprocess.call(command)

	nThread = myThread(lines[i], i)
	nThread.start()
	threads.append(nThread)

for t in threads:
	t.join()

os.system("rm full*")
os.system("rm *.webm")
os.system("mv *.mp4 Video/")
os.system("rm *.part")