import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import sys
import os

def run_model(X_train_filepath, X_test_filepath, y_train_filepath, y_test_filepath, learning_rate, batch_size, epochs):
	X_train = np.load(X_train_filepath)
	X_test = np.load(X_test_filepath)
	y_train = np.load(y_train_filepath)
	y_test = np.load(y_test_filepath)

	model_number = os.listdir('./logs')
	model_number = [int(model.split('_')[1]) for model in model_number]
	model_number = str(max(model_number) + 1) + '_image'

	# create an empty network model
	model = Sequential()

	# --- input layer ---
	model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
	# --- max pool ---
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))		
	
	# flatten for fully connected classification layer
	model.add(Flatten())
	# note that the 10 is the number of classes we have
	# the classes are mutually exclusive so softmax is a good choice
	# --- fully connected layer ---
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	# --- classification ---
	model.add(Dense(y_train.shape[1], activation='softmax'))

	# prints out a summary of the model architecture
	print(model.summary())

	adam = Adam(lr=float(learning_rate))
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	
	reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, min_lr = 0.00001)
	early_stopping = EarlyStopping(monitor='val_loss', patience = 10)
	tensorboard = TensorBoard(log_dir='logs/model_'+model_number, histogram_freq=1, write_graph=True, write_images=False)

	history = model.fit(X_train, y_train, class_weight='auto', validation_data=(X_test, y_test), batch_size=int(batch_size), epochs=int(epochs), verbose=1, callbacks = [reduce_lr, tensorboard, early_stopping])

	# once training is complete, let's see how well we have done
	train_predictions = model.predict(X_train)
	test_predictions = model.predict(X_test)

	np.save('predictions/model_' + model_number + '_train_predictions', train_predictions)
	np.save('predictions/model_' + model_number + '_test_predictions', test_predictions)

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

X_train_filepath = sys.argv[1]
X_test_filepath = sys.argv[2]
y_train_filepath = sys.argv[3]
y_test_filepath = sys.argv[4]
learning_rate = sys.argv[5]
batch_size = sys.argv[6]
epochs = sys.argv[7]
run_model(X_train_filepath, X_test_filepath, y_train_filepath, y_test_filepath, learning_rate, batch_size, epochs)