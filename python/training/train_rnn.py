# This program trains the rnn that will trade stocks

import numpy as np
import pandas as pd


def prepare(trainset_path, NUMBARS, TRAINBARLENGTH):
	# Read in the dataset and save as panda dataframe
	dataset_train = pd.read_csv(trainset_path, sep=r'\s*,\s*', engine='python')
	# convert panda dataframe to numpy array
	training_set = dataset_train.to_numpy()
	#print("Dataset Head: ")
	#print(dataset_train.head())


	# Normalize data
	from sklearn.preprocessing import MinMaxScaler
	sc = MinMaxScaler(feature_range=(0,1))
	training_set_scaled = sc.fit_transform(training_set)


	# Convert dataset into x_train with shape of (Number of tuples, NUMBARS, data per bar)
	# y_train being shape of (Number of tuples,)
	# Data per bar being price, volume, etc
	# x_train is data that is being trained on
	# y_train is considered the true value that the network should get
	x_train = []
	y_train = []
	for i in range(NUMBARS, TRAINBARLENGTH):
	  x_train.append(training_set_scaled[i-NUMBARS:i])
	  y_train.append(training_set_scaled[i, 2])
	x_train, y_train = np.array(x_train), np.array(y_train)
	return x_train, y_train


"""Build Neural Network"""
def build_network(NUMBARS):
	import keras
	from keras.models import Sequential
	from keras.layers import LSTM, Dropout, Dense, Flatten

	model = Sequential()

	model.add(LSTM(units=50,return_sequences=True, input_shape=(NUMBARS, 5)))
	model.add(Dropout(0.2))

	model.add(LSTM(units=50,return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(units=50,return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(units=50,))
	model.add(Dropout(0.2))

	model.add(Dense(units=1))

	#print ('Model: ')
	#model.summary()
	model.compile(optimizer='adam', loss= 'mean_squared_error')
	return model
	
def train_network(x_train, y_train, num_epochs, model):
	model.fit(x_train,y_train,epochs=num_epochs, batch_size=32)
	return model

"""Test the Results"""

def predict(trainset_path, testset_path, model, NUMBARS):
	dataset_train = pd.read_csv(trainset_path, sep=r'\s*,\s*', engine='python')
	# convert panda dataframe to numpy array
	training_set = dataset_train.to_numpy()
	from sklearn.preprocessing import MinMaxScaler
	sc = MinMaxScaler(feature_range=(0,1))
	training_set_scaled = sc.fit_transform(training_set)
	
	# Read in test dataset
	#print('Testing...')
	dataset_test = pd.read_csv(testset_path, sep=r'\s*,\s*', engine='python')
	real_stock_price = dataset_test.to_numpy()

	# Reshape data
	inputs = sc.transform(dataset_test[:].values)
	X_test = []
	for i in range(NUMBARS, 1000):
	  X_test.append(inputs[i-NUMBARS:i])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],  5))

	# Predict next prices
	predicted_stock_price = model.predict(X_test)

	# Add 4 columns of 0 onto predictions so it can be fed back through sc
	shaped_predictions = np.empty(shape = (990, 5))
	for row in range(0, 990):
		shaped_predictions[row, 0] = predicted_stock_price[row, 0]
		for col in range (1, 5):
			shaped_predictions[row, col] = 0
		
	# un-Normalize data
	predicted_stock_price = sc.inverse_transform(shaped_predictions)
	
	return real_stock_price, predicted_stock_price
	
def validate_results(trainset_path, testset_path, model, NUMBARS):
	real_price, predicted_price = predict(trainset_path, testset_path, model, NUMBARS)

	# calculate error
	error_sum = 0
	for i in range(0, len(real_price) - NUMBARS):
		error = abs(real_price[i,2] - predicted_price[i,0])
		error_sum += error
	return error_sum
	
def test_results(trainset_path, testset_path, model, NUMBARS):
	real_price, predicted_price = predict(trainset_path, testset_path, model, NUMBARS)
	
	# Graph results
	print('Results: ')
	import matplotlib
	'''
	import os
	
	if not os.path.isdir('data/results/stock' + str(stock) + '/'):
		os.mkdir('data/results/stock' + str(stock) + '/')
	'''
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
	plt.plot(real_price[:,2], color = 'black', label = 'Stock Price')
	plt.plot(predicted_price[:,0], color = 'green', label = 'Price Prediction')
	plt.title('Prediction')
	plt.xlabel('Time')
	plt.ylabel('Stock Price')
	plt.legend()
	#plt.savefig('data/results/stock' + str(stock) + '/training' + str(ID) + '.png')
	plt.show()
	#plt.close()
	
	'''
	print('Error is: ' + str(error))
	error_sheet = open(r"data/results/error.csv",'a')
	error_sheet.write(str(error))
	error_sheet.close()
	
	error_sheet = open(r"data/results/error.csv",'a')
		error_sheet.write(str(current_stock) + ",")
		error_sheet.close()
			
		for stock in range(what_stocks[2], what_stocks[3]):
			self._train_collect(stock, 'test', User)
			
			
			tr.test_results(TRAINSET, TESTSET, model, NUMBARS, stock, current_stock)
		
			error_sheet = open(r"data/results/error.csv",'a')
			error_sheet.write(",")
			error_sheet.close()
			
		error_sheet = open(r"data/results/error.csv",'a')
		error_sheet.write("\n")
		error_sheet.close()	
	'''



	