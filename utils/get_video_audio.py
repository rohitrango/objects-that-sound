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
	def __init__(self, lin, count, isValString):
		threading.Thread.__init__(self)
		self.lin = lin
		self.count = count
		self.isValString = isValString

	def run(self):
		print ("Starting " + self.name)
		download_vid(self.lin, self.count, self.isValString)
		print ("Exiting " + self.name)

vid2class = dict()


def checkBalanceInFiles(path="Video/"):
	# Load all classes
	with open('metadata/tags.cls') as fi:
		classes = dict(map(lambda x: (x[:-1], 0), fi.readlines()))
	
	# Load the lines from the csv
	fileToClassMap = dict()
	with open('metadata/videos.csv') as fi:
		lines = fi.readlines()
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			video_id = words[0]
			fileToClassMap[video_id] = words[3]

	# Get all downloaded files
	for root, dirs, files in os.walk(path):
		break

	files = map(lambda x: x[6:-4], files)
	for video_id in files:
		for xcls in fileToClassMap[video_id]:
			classes[xcls] += 1

	return classes


def filtValidation():

	# Load all classes
	with open('metadata/tags.cls') as fi:
		classes = dict(map(lambda x: (x[:-1], 0), fi.readlines()))

	checkVideo = []
	fileToClassMap = dict()		
	with open('csv/balanced_train_segments_filtered.csv') as fi:
		lines = fi.readlines()
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			video_id = words[0]

			# Add potential video and its details into checkVideo and fileToClass map

			checkVideo.append((words[0], lin))
			fileToClassMap[video_id] = words[3]			

	
	for r, dirs, files in os.walk("Video/"):
		if len(files) > 0:
			break

	# filter files to only have the video_id
	files = map(lambda x: x.split(".mp4")[0].split("video_")[1], files)
	
	# CheckVideo is the list of videos not downloaded with their details in fileToClassMap
	checkVideo = filter(lambda x: x[0] not in files, checkVideo)
	for video_id, lin in checkVideo:
		for xcls in fileToClassMap[video_id]:
			classes[xcls] += 1

	
	# Open a new file and dump it there
	with open("csv/balanced_validation.csv", "w+") as fi:
		fi.writelines(map(lambda x: x[1], checkVideo))

	print("Done")



def create_unbalanced_files(lines, filename='csv/unbalanced_train_segments_filtered.csv'):
	with open(filename, 'w') as fi:
		lines = fi.readlines()
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			newtags = list(set(tags)&set(words[-1]))
			if len(newtags) > 0:
				fi.write(words[0]+","+words[1]+","+words[2]+","+",".join(newtags)+"\n")


def download_vid(lin, count, valString):

	# Extract the words consisting of video_id, start_time, end_time, list of video_tags
	words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
	words = words[0:3] + [words[3:]]
	video_id = words[0]

	if os.path.exists("Video" + valString + "/video_" + video_id + ".mp4"):
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


def convertAudio(path="."):

	for r, dirs, files in os.walk(path):
		break

	for f in files:
		audio_file_name = f.replace("video_", "audio_").replace(".mp4", ".wav")
		command = ["ffmpeg", "-i", f,"-ab","160k", "-ac","1","-ar","48000","-vn",audio_file_name]
		subprocess.call(command)
	print("Done")


def downloadAllVideos(validation=False):
	# Lines for every video
	print("Validation : {0}".format(validation))
	if validation:
		filename = "csv/balanced_validation.csv"
		isValString = "_val"
	else:
		filename = "csv/unbalanced_train_segments_filtered.csv"
		isValString = ""

	with open(filename) as f:
		lines = f.readlines()

	print("Downloading {0} videos.".format(len(lines)))
	# Load all tags for checking download
	with open('metadata/tags.cls') as file:
		tags = map(lambda x: x[:-1], file.readlines())

	print(tags)

	threads = []
	start = 8002
	# lines.reverse()

	for i in range(len(lines[start:])):

		if len(threads) == 2:
			for t in threads:
				t.join()
				print "Joined thread"
			threads = []
			print "Joined Threads"
			os.system("rm full*")
			# os.system("rm *.webm")
			os.system("mv *.mp4 Video{0}/".format(isValString))
			os.system("mv *.wav Audio{0}/".format(isValString))

			# subprocess.call(command)
			# command = ["rm", "*.webm"]
			# subprocess.call(command)

		nThread = myThread(lines[i+start], i+start, isValString)
		nThread.start()
		threads.append(nThread)

	for t in threads:
		t.join()

	os.system("rm full*")
	os.system("rm *.webm")
	os.system("mv *.mp4 Video{0}/".format(isValString))
	os.system("mv *.wav Audio{0}/".format(isValString))
	os.system("rm *.part")


if __name__ == "__main__":
	downloadAllVideos(False)

