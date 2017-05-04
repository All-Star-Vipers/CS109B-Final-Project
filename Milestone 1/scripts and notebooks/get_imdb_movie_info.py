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
	#try:
	values = ia.get_movie(movie_id)
	print(len(values.keys()))
	#except:
	#	print("Error")
	#	values = None
	print("{0} out of {1} | {2:.2f}% Complete".format(index, total, index*1./total * 100))
	return values

def get_data(tmdb_filepath, imdb_filepath):
	df_popular = pd.read_csv(tmdb_filepath)
	try:
		df = pd.read_table(imdb_filepath, sep = ',', error_bad_lines = False)
		ids_to_pull = list(set(df_popular['imdb_id']) - set(df['imdb_id']))
	except IOError:		
		df = pd.DataFrame({})
		ids_to_pull = df_popular['imdb_id'].tolist()
	total = len(ids_to_pull)
	counter = 1
	values_to_list = ['smart canonical episode title', 'canonical episode title', 'episode title', 'long imdb episode title', 'canonical title', 'long imdb canonical title', 'long imdb title', 'smart canonical title', 'smart long imdb canonical title', 'title', 'plot outline']
	people_keys = ['producer', 'director', 'cinematographer', 'miscellaneous crew','production companies', 'writer', 'cast', 'editor']
	while len(ids_to_pull) > 0:
		movie_id = ids_to_pull[0]
		start_time = time.time()
		values = get_response(movie_id, counter , total)
		print(values)
		imdb_dict = {}
		if values:
			for key in values.keys():
				if key in people_keys:
					imdb_dict[key] = [item['name'] for item in values[key]]
				else:
					imdb_dict[key] = values[key]
			imdb_dict['imdb_id'] = movie_id
			imdb_dict['movie_status_code'] = True
			for value in values_to_list:
				if value in values.keys():
					imdb_dict[value] = [imdb_dict[value]]
		else:
			imdb_dict = {'imdb_id' : movie_id}
		df = df.append([imdb_dict])
		try:
			df.to_csv(imdb_filepath, index =False)
		except:
			for column in df.columns:
				print(column)
				df[column].astype(str)
			break
		ids_to_pull.remove(movie_id)
		counter = counter + 1
		end_time = time.time()
		time.sleep(max(.25 - (end_time - start_time), 0))
	print(df['movie_status_code'].value_counts())
	print(df['movie_status_code'].isnull().sum())
	print("Complete")

tmdb_filepath = sys.argv[1]
imdb_filepath = sys.argv[2]
get_data(tmdb_filepath, imdb_filepath)