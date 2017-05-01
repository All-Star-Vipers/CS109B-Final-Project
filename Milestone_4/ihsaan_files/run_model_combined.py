import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Merge, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from sklearn.preprocessing import StandardScaler
import sys
import os

def run_model(X_train_image_filepath, X_test_image_filepath, X_train_metadata_filepath, X_test_metadata_filepath, y_train_filepath, y_test_filepath, learning_rate, batch_size, epochs):
	X_train_image = np.load(X_train_image_filepath)
	X_test_image = np.load(X_test_image_filepath)
	X_train_metadata = np.load(X_train_metadata_filepath)
	X_test_metadata = np.load(X_test_metadata_filepath)
	y_train = np.load(y_train_filepath)
	y_test = np.load(y_test_filepath)

	scaler = StandardScaler()
	X_train_metadata = scaler.fit_transform(X_train_metadata)
	X_test_metadata = scaler.transform(X_test_metadata)

	model_number = os.listdir('./logs')
	model_number = [int(model.split('_')[1]) for model in model_number]
	model_number = str(max(model_number) + 1) + '_combined'

	# use CNN for images
	image_branch = Sequential()
	image_branch.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=X_train_image.shape[1:]))
	image_branch.add(MaxPooling2D(pool_size=(2, 2)))
	image_branch.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
	image_branch.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
	image_branch.add(MaxPooling2D(pool_size=(2, 2)))
	image_branch.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	image_branch.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	image_branch.add(MaxPooling2D(pool_size=(2, 2)))
	image_branch.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	image_branch.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	image_branch.add(MaxPooling2D(pool_size=(2, 2)))	
	image_branch.add(Flatten())
	image_branch.add(Dense(512, activation='relu'))
	image_branch.add(Dropout(0.5))
	image_branch.add(Dense(64, activation='relu'))
	image_branch.add(Dropout(0.5))

	# use MLP for metadata
	metadata_branch = Sequential()
	metadata_branch.add(Dense(8, activation='relu', input_shape=(X_train_metadata.shape[1],)))
	metadata_branch.add(Dense(16, activation='relu'))
	metadata_branch.add(Dense(32, activation='relu'))
	metadata_branch.add(Dense(64, activation='relu'))
	metadata_branch.add(Dense(128, activation='relu'))
	metadata_branch.add(Dense(256, activation='relu'))
	metadata_branch.add(Dropout(0.5))
	metadata_branch.add(Dense(512, activation='relu'))
	metadata_branch.add(Dropout(0.5))

	# merge models
	model = Sequential()
	model.add(Merge([image_branch, metadata_branch], mode = 'concat'))
	# --- classification ---
	model.add(Dense(y_train.shape[1], activation='softmax'))

	# prints out a summary of the model architecture
	print(model.summary())

	adam = Adam(lr=float(learning_rate))
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	
	reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 10, min_lr = 0.00001)
	early_stopping = EarlyStopping(monitor='val_loss', patience = 50)
	tensorboard = TensorBoard(log_dir='logs/model_'+model_number, histogram_freq=1, write_graph=True, write_images=False)

	history = model.fit([X_train_image, X_train_metadata], y_train, class_weight='auto', validation_data=([X_test_image, X_test_metadata], y_test), batch_size=int(batch_size), epochs=int(epochs), verbose=1, callbacks = [reduce_lr, tensorboard, early_stopping])

	# once training is complete, let's see how well we have done
	train_predictions = model.predict([X_train_image, X_train_metadata])
	test_predictions = model.predict([X_test_image, X_test_metadata])

	np.save('predictions/model_' + model_number + '_train_predictions', train_predictions)
	np.save('predictions/model_' + model_number + '_test_predictions', test_predictions)

	score = model.evaluate([X_test_image, X_test_metadata], y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

X_train_image_filepath = sys.argv[1]
X_test_image_filepath = sys.argv[2]
X_train_metadata_filepath = sys.argv[3]
X_test_metadata_filepath = sys.argv[4]
y_train_filepath = sys.argv[5]
y_test_filepath = sys.argv[6]
learning_rate = sys.argv[7]
batch_size = sys.argv[8]
epochs = sys.argv[9]
run_model(X_train_image_filepath, X_test_image_filepath, X_train_metadata_filepath, X_test_metadata_filepath, y_train_filepath, y_test_filepath, learning_rate, batch_size, epochs)