# coding: utf-8
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import json
import youtube_dl
import subprocess
import time

counter = 0

with open("ontology.json") as f:
	data = json.load(f)
	
classes = dict([ (str(x['id']), str(x['name'])) for x in data])

with open("balanced_train_segments.csv") as f:
	lines = f.readlines()

vid2class = dict()
for lin in lines:
	words =  [word.replace(" ","") for word in lin.split(",")]
	words = words[0:3] + [[word.replace('"','').replace("\n",'') for word in words[3:]]]

	vid2class[words[0]]=words[3]

	ydl_opts = {
		'start_time': int(float(words[1])),
		'end_time': int(float(words[2])),
		'format': 'best[height<=360]',
		'outtmpl' : r"{}_{}.%(ext)s".format("video", counter)
	}

	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		ydl.download(['https://www.youtube.com/watch?v=' + words[0]])
		info = ydl.extract_info(str('https://www.youtube.com/watch?v=' + words[0]), download=False)
		ext = info.get('ext',None)
		ps = subprocess.Popen(('ls', '-1tc'), stdout=subprocess.PIPE)
		output = subprocess.check_output(('head', '-n1'), stdin=ps.stdout)
		ps.wait()
		# print(output)
		# terminal.call(["ls","-t","|","head","-n1"])
		print("00:00:"+str(int(float(words[1]))))
		subprocess.call(["ffmpeg","-ss",str(int(float(words[1]))),"-i","video_"+str(counter)+"."+ext,"-t","00:00:10","-vcodec","copy","-acodec","copy","new_video_"+str(counter)+"."+ext])
		counter = counter + 1
		# ffmpeg -i new_video_1.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav
		# print("File name is "+title)

	print("Im Done")