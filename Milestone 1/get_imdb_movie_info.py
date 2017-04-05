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
	values = ia.get_movie(str(movie_id))
	print(len(values.keys()))
	#except:
	#	print("Error")
	#	values = None
	print("{0} out of {1} | {2:.2f}% Complete".format(index, total, index*1./total * 100))
	return values

def get_data(popular_movies_filepath, detailed_movies_filepath):
	df_popular = pd.read_csv(popular_movies_filepath)
	try:
		df = pd.read_csv(detailed_movies_filepath)
		ids_to_pull = list(set(df_popular['imdb_id']) - set(df['imdb_id']))
	except IOError:		
		df = pd.DataFrame({})
		ids_to_pull = df_popular['imdb_id'].tolist()
	total = len(ids_to_pull)
	counter = 1
	values_to_list = ['canonical title', 'long imdb canonical title', 'long imdb title', 'smart canonical title', 'smart long imdb canonical title', 'title']
	while len(ids_to_pull) > 0:
		movie_id = ids_to_pull[0]
		start_time = time.time()
		status_code, values = get_response(movie_id, counter , total)
		if values:
			values['imdb_id'] = movie_id
			values['movie_status_code'] = True
			for value in values_to_list:
				values[value] = [values[value]]
		else:
			values = {'imdb_id' : movie_id}
		df = df.append([values])
		#for column in df.columns:
		#	print(column)
		#	print(values[column])
		df.to_csv(detailed_movies_filepath, index =False)
		ids_to_pull.remove(movie_id)
		counter = counter + 1
		end_time = time.time()
		time.sleep(max(.75 - (end_time - start_time), 0))
	print(df['movie_status_code'].value_counts())
	print(df['movie_status_code'].isnull().sum())
	print("Complete")

popular_movies_filepath = sys.argv[1]
detailed_movies_filepath = sys.argv[2]
get_data(popular_movies_filepath, detailed_movies_filepath)