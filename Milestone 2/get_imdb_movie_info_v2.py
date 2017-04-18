import pandas as pd
import numpy as np
import json
import sys
import requests
import time
from ast import literal_eval
from imdb import IMDb

def get_response(movie_id, index, total):
	ia = IMDb()
	values = ia.get_movie(movie_id)
	print(len(values.keys()))
	print("{0} out of {1} | {2:.2f}% Complete".format(index, total, index*1./total * 100))
	return values

def get_data(tmdb_filepath, imdb_filepath):
	df_popular = pd.read_csv(tmdb_filepath)
	try:
		df = pd.read_csv(imdb_filepath)
		ids_to_pull = list(set(df_popular['imdb_id']) - set(df['imdb_id']))
	except IOError:		
		df = pd.DataFrame({})
		ids_to_pull = df_popular['imdb_id'].tolist()
	total = len(ids_to_pull)
	counter = 1
	departments = ['animation department', 'art department', 'camera and electrical department', 'cast', 'casting department','costume department', 'distributors',
	'editorial department', 'makeup', 'music department', 'special effect department', 'visual effects']
	text = ['mpaa', 'plot outline', 'title']
	numbers = ['rating','votes', 'genres', 'plot']
	while len(ids_to_pull) > 0:
		movie_id = ids_to_pull[0]
		#start_time = time.time()
		values = get_response(movie_id, counter , total)
		print(values)
		imdb_dict = {}
		if values:
			imdb_dict['movie_status_code'] = True
			imdb_dict['imdb_id'] = movie_id
			for department in departments: 
				if department in values.keys():
					imdb_dict[department] = len(values[department])
			for txt in text:
				if txt in values.keys():
					imdb_dict[txt] = [values[txt]]
			for number in numbers:
				if number in values.keys():
					imdb_dict[number] = values[number]
		else:
			imdb_dict['imdb_id'] = movie_id
			imdb_dict['movie_status_code'] = False
		df = df.append([imdb_dict])
		df.to_csv(imdb_filepath, index =False)
		ids_to_pull.remove(movie_id)
		counter = counter + 1
		#end_time = time.time()
		#time.sleep(max(.25 - (end_time - start_time), 0))
	print(df['movie_status_code'].value_counts())
	print(df['movie_status_code'].isnull().sum())
	print("Complete")

tmdb_filepath = sys.argv[1]
imdb_filepath = sys.argv[2]
get_data(tmdb_filepath, imdb_filepath)
