# coding: utf-8
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import json
import youtube_dl
import subprocess
import time
import threading

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

def download_vid(lin,count):
	words =  [word.replace(" ","") for word in lin.split(",")]
	words = words[0:3] + [[word.replace('"','').replace("\n",'') for word in words[3:]]]
	video_id = words[0]

	vid2class[video_id]=words[3]

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


with open("ontology.json") as f:
	data = json.load(f)
	
classes = dict([ (str(x['id']), str(x['name'])) for x in data])

with open("balanced_train_segments.csv") as f:
	lines = f.readlines()

threads = []
i = 0

while i<len(lines):
	print i
	try:
		thread1 = myThread(lines[i], i)
		thread2 = myThread(lines[i+1], i+1)
		thread3 = myThread(lines[i+2], i+2)

		thread1.start()
		thread2.start()
		thread3.start()

		threads.append(thread1)
		threads.append(thread2)
		threads.append(thread3)

	except:
		print "Error: unable to start thread"
	i += 3

	for t in threads:
		t.join()
		print "Joined thread"
	print "Joined Threads"
	threads = []