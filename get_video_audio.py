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
		download_vid(self.lin, self.count)
		print ("Exiting " + self.name)

vid2class = dict()


def checkBalanceInFiles():
	# Load all classes
	with open('tags.cls') as fi:
		classes = dict(map(lambda x: (x[:-1], 0), fi.readlines()))
	
	# Load the lines from the csv
	fileToClassMap = dict()
	with open('unbalanced_train_segments_filtered.csv') as fi:
		lines = fi.readlines()
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			video_id = words[0]
			fileToClassMap[video_id] = words[3]


	with open('balanced_train_segments_filtered.csv') as fi:
		lines = fi.readlines()
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			video_id = words[0]
			fileToClassMap[video_id] = words[3]

	# Get all downloaded files
	for root, dirs, files in os.walk('Video/'):
		break

	files = map(lambda x: x[6:-4], files)
	for video_id in files:
		for xcls in fileToClassMap[video_id]:
			classes[xcls] += 1

	return classes


def create_unbalanced_files(lines, filename='unbalanced_train_segments_filtered.csv'):
	with open(filename, 'w') as fi:
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			newtags = list(set(tags)&set(words[-1]))
			if len(newtags) > 0:
				fi.write(words[0]+","+words[1]+","+words[2]+","+",".join(newtags)+"\n")


def download_vid(lin, count):

	# Extract the words consisting of video_id, start_time, end_time, list of video_tags
	words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
	words = words[0:3] + [words[3:]]
	video_id = words[0]

	if os.path.exists("Video/video_" + video_id + ".mp4"):
		print("File already exists.")
		return None
	else:
		print("File doesn't exist.")

	vid2class[video_id] = words[3]

	ydl_opts = {
		'start_time': int(float(words[1])),
		'end_time': int(float(words[2])),
		'format': 'mp4[height<=360]',
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
		command = ["ffmpeg", "-i", "video_"+video_id+"."+ext,"-ab","160k", "-ac","1","-ar","48000","-vn",audio_file_path]
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
start = 0

for i in range(len(lines[start:])):

	if len(threads) == 2:
		for t in threads:
			t.join()
			print "Joined thread"
		threads = []
		print "Joined Threads"
		os.system("rm full*")
		# os.system("rm *.webm")
		os.system("mv *.mp4 Video/")
		os.system("mv *.wav Audio/")

		# subprocess.call(command)
		# command = ["rm", "*.webm"]
		# subprocess.call(command)

	nThread = myThread(lines[i+start], i+start)
	nThread.start()
	threads.append(nThread)

for t in threads:
	t.join()

os.system("rm full*")
os.system("rm *.webm")
os.system("mv *.mp4 Video/")
os.system("rm *.part")