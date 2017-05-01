import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
import sys
import os

def run_model(X_train_filepath, X_test_filepath, y_train_filepath, y_test_filepath, learning_rate, batch_size, epochs):
	X_train = np.load(X_train_filepath)
	X_test = np.load(X_test_filepath)
	y_train = np.load(y_train_filepath)
	y_test = np.load(y_test_filepath)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	model_number = os.listdir('./logs')
	model_number = [int(model.split('_')[1]) for model in model_number]
	model_number = str(max(model_number) + 1) + '_metadata'

	# create an empty network model
	model = Sequential()
	model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	
	# --- classification ---
	model.add(Dense(y_train.shape[1], activation='softmax'))

	# prints out a summary of the model architecture
	print(model.summary())

	adam = Adam(lr=float(learning_rate))
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	
	reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 10, min_lr = 0.00001)
	early_stopping = EarlyStopping(monitor='val_loss', patience = 20)
	tensorboard = TensorBoard(log_dir='logs/model_'+model_number, histogram_freq=1, write_graph=True, write_images=False)

	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=int(batch_size), epochs=int(epochs), verbose=1, callbacks = [tensorboard, reduce_lr, early_stopping])

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