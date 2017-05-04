import pandas as pd
import numpy as np
import json
import sys
import requests
import time
from ast import literal_eval

def get_response(movie_id, index, total):
	api_token = '9ec0c2e7850f575e7dcd37c195e45b69'
	request_url = 'https://api.themoviedb.org/3/movie/{0}?api_key={1}'.format(movie_id, api_token)
	try:
		response = requests.get(request_url)
		status_code = response.status_code
		data = json.loads(response.text)
		print(status_code)
		values = data
	except:
		print("Error")
		values = None
	print("{0} out of {1} | {2:.2f}% Complete".format(index, total, index*1./total * 100))
	return (status_code, values)

def get_data(popular_movies_filepath, detailed_movies_filepath):
	df_popular = pd.read_csv(popular_movies_filepath)
	try:
		df = pd.read_csv(detailed_movies_filepath)
		ids_to_pull = list(set(df_popular['id']) - set(df['id']))
	except IOError:		
		df = pd.DataFrame({})
		ids_to_pull = df_popular['id'].tolist()
	total = len(ids_to_pull)
	counter = 1
	while len(ids_to_pull) > 0:
		movie_id = ids_to_pull[0]
		start_time = time.time()
		status_code, values = get_response(movie_id, counter , total)
		if values and status_code != 404:
			values['id'] = movie_id
			values['movie_status_code'] = 200
			values['overview'] = [values['overview']]
			values['title'] = [values['title']]
			values['original_title'] = [values['original_title']]
			values['tagline'] = [values['tagline']]
			values['homepage'] = [values['homepage']]
		else:
			values = {'id' : movie_id}
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