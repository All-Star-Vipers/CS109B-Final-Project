import pandas as pd
import numpy as np
from PIL import Image
from scipy import ndimage, misc
import  sys
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

POSTER_PREFIX_PATH = 'posters/'

def create_keras_image_array(df, img_height, img_width):
	poster_array = []
	for i, value in df['poster'].iteritems():
		if len(value.shape) < 3:
			value = np.resize(value, (value.shape[0], value.shape[1], 3))
		poster_array.append(value)
	return np.asarray(poster_array)

def create_keras_metadata_array(df_original, df_poster):
	df_poster = df_poster.drop('genre', axis = 1)
	df_combined = df_poster.merge(df_original, how = 'left', on = 'id')
	df_combined = df_combined.drop(['poster', 'genre', 'id', 'poster_path', 'pp'], axis = 1)
	df_combined = df_combined.astype(float)
	return df_combined.as_matrix()

def encode_outcomes(genres):
	encoder = LabelEncoder()
	encoded_Y = encoder.fit_transform(genres)
	y = np_utils.to_categorical(encoded_Y)
	return y

def convert_jpg_array(poster_path, img_height, img_width):
    '''
    inputs: director of poster
    outputs: array of poster with given heights and width
    '''
    try:
    	img = Image.open(poster_path)
    	img = misc.imresize(img, (img_height, img_width))
    	arr = np.array(img)
    except:
    	arr = None
    return arr

def convert_image_to_matrix(train_filepath, test_filepath, img_height, img_width):
	train_df = pd.read_csv(train_filepath, encoding = "ISO-8859-1")
	test_df = pd.read_csv(test_filepath, encoding = "ISO-8859-1")
	train_df = train_df[train_df['poster_path'].notnull()]
	test_df = test_df[test_df['poster_path'].notnull()]
	
	train_df['poster_path'] = train_df['poster_path'].apply(lambda path: POSTER_PREFIX_PATH + str(path) + '.jpg')
	test_df['poster_path'] = test_df['poster_path'].apply(lambda path: POSTER_PREFIX_PATH  + str(path) + '.jpg')

	poster_train = train_df['poster_path'].apply(lambda x: convert_jpg_array(x, img_height, img_width))
	poster_test = test_df['poster_path'].apply(lambda x: convert_jpg_array(x, img_height, img_width))

	# train - recombine genres and posters, filtering out any null poster values
	poster_train_df = train_df[['genre', 'id']]
	poster_train_df['poster'] = poster_train
	poster_train_df = poster_train_df[poster_train_df['poster'].notnull()]
	#remove  weird 4-d array
	poster_train_df['poster_dimension'] = poster_train_df['poster'].apply(lambda x: x.shape[2] if len(x.shape) == 3 else 3)
	poster_train_df = poster_train_df[poster_train_df['poster_dimension'] == 3]
	poster_train_df = poster_train_df.drop('poster_dimension', axis = 1)

	# test - recombine genres and posters, filtering out any null poster values
	poster_test_df = test_df[['genre', 'id']]
	poster_test_df['poster'] = poster_test
	poster_test_df = poster_test_df[poster_test_df['poster'].notnull()]
	#remove  weird 4-d array
	poster_test_df['poster_dimension'] = poster_test_df['poster'].apply(lambda x: x.shape[2] if len(x.shape) == 3 else 3)
	poster_test_df = poster_test_df[poster_test_df['poster_dimension'] == 3]
	poster_test_df = poster_test_df.drop('poster_dimension', axis = 1)

	print('Pre-Encode Train Shape {0}'.format(train_df.shape))
	print('Post-Encode Train Shape {0}'.format(poster_train_df.shape))
	print('Pre-Encode Test Shape {0}'.format(test_df.shape))
	print('Post-Encode Test Shape {0}'.format(poster_test_df.shape))

	X_train_images = create_keras_image_array(poster_train_df, img_height, img_width)
	X_test_images = create_keras_image_array(poster_test_df, img_height, img_width)

	print('X Train Images Shape: {0}'.format(X_train_images.shape))
	print('X Test Images Shape: {0}'.format(X_test_images.shape))

	X_train_metadata = create_keras_metadata_array(train_df, poster_train_df)
	X_test_metadata = create_keras_metadata_array(test_df, poster_test_df)

	print('X Train Metadata Shape: {0}'.format(X_train_metadata.shape))
	print('X Test Metadata Shape: {0}'.format(X_test_metadata.shape))

	y_train = encode_outcomes(poster_train_df['genre'])
	y_test = encode_outcomes(poster_test_df['genre'])

	print('Y Train Shape: {0}'.format(y_train.shape))
	print('Y Test Shape: {0}'.format(y_test.shape))

	np.save('train_predictor_image_keras', X_train_images)
	np.save('train_predictor_metadata_keras', X_train_metadata)
	np.save('train_outcome_keras', y_train)
	np.save('test_predictor_image_keras', X_test_images)
	np.save('test_predictor_metadata_keras', X_test_metadata)
	np.save('test_outcome_keras', y_test)

	print('Complete')

train_filepath = sys.argv[1]
test_filepath = sys.argv[2]
img_height = int(sys.argv[3])
img_width = int(sys.argv[4])
convert_image_to_matrix(train_filepath, test_filepath, img_height, img_width)