import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
import sys

def run_model(X_train_filepath, X_test_filepath, y_train_filepath, y_test_filepath, tb_marker, learning_rate, batch_size, epochs):
	X_train = np.load(X_train_filepath)
	X_test = np.load(X_test_filepath)
	y_train = np.load(y_train_filepath)
	y_test = np.load(y_test_filepath)

	# create an empty network model
	model = Sequential()

	# --- input layer ---
	model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
	# --- max pool ---
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# flatten for fully connected classification layer
	model.add(Flatten())
	# note that the 10 is the number of classes we have
	# the classes are mutually exclusive so softmax is a good choice
	# --- fully connected layer ---
	model.add(Dense(16, activation='relu'))
	# --- classification ---
	model.add(Dense(y_train.shape[1], activation='softmax'))

	# prints out a summary of the model architecture
	print(model.summary())

	adam = Adam(lr=float(learning_rate))
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, min_lr = 0.001)

	tensorboard = TensorBoard(log_dir='logs/model_'+tb_marker, histogram_freq=1, write_graph=True, write_images=False)

	history = model.fit(X_train, y_train, validation_split = 0.2, batch_size=int(batch_size), epochs=int(epochs), verbose=1, callbacks = [reduce_lr, tensorboard])

	# once training is complete, let's see how well we have done
	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

X_train_filepath = sys.argv[1]
X_test_filepath = sys.argv[2]
y_train_filepath = sys.argv[3]
y_test_filepath = sys.argv[4]
tensorboard_marker = sys.argv[5]
learning_rate = sys.argv[6]
batch_size = sys.argv[7]
epochs = sys.argv[8]
run_model(X_train_filepath, X_test_filepath, y_train_filepath, y_test_filepath, tensorboard_marker, learning_rate, batch_size, epochs)