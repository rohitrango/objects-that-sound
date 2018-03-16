# coding: utf-8
import numpy as np
import pandas as pd
import json
import youtube_dl

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
		'format': 'worst',
	}

	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		ydl.download(['https://www.youtube.com/watch?v=' + words[0]])

	print("Im Done")