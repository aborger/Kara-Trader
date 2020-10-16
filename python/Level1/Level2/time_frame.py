import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Time_frame:
	_frames = []
	_NUMBARS = 0
	_model = 0
	_api = 0
	# time_frame can be 1Min, 5Min, 15Min, or 1D
	def __init__(self, frame, symbol):
		self.frame_name = frame
		self.symbol = symbol
		self._frames.append(self)
	

	def get_current_price(self):
		barset = (Time_frame._api.get_barset(self.symbol,'1Min',limit=1))
		symbol_bars = barset[self.symbol]
		current_price = symbol_bars[0].c
		return current_price
	
	def get_gain(self):
		prediction = self.get_prediction()
		current = self.get_current_price()
		
		gain = prediction/current
		gain = round((gain -1) * 100, 3)
		if gain < 0:
			gain = 0
		self.gain = gain
	
	def get_normalized_data(self):
		# Get bars
		barset = Time_frame._api.get_barset(self.symbol,self.frame_name,limit=Time_frame._NUMBARS)
		
		# Get symbol's bars
		symbol_bars = barset[self.symbol]

		# Convert to list
		dataSet = []

		for barNum in symbol_bars:
			bar = []
			bar.append(barNum.o)
			bar.append(barNum.c)
			bar.append(barNum.h)
			bar.append(barNum.l)
			bar.append(barNum.v)
			dataSet.append(bar)
			
		  
		# Convert to numpy array
		npDataSet = np.array(dataSet)
		reshapedSet = np.reshape(npDataSet, (1, Time_frame._NUMBARS, 5))
		# Normalize Data
		sc = MinMaxScaler(feature_range=(0,1))
		normalized = np.empty(shape=(1, Time_frame._NUMBARS, 5)) 
		normalized[0] = sc.fit_transform(reshapedSet[0])
		return normalized
		
	def get_prediction(self):
		#print('Getting prediction for ' self.symbol ' on ' + time_frame + ' time frame')

		normalized = get_normalized_data()

		
		# Predict Price
		predicted_price = Time_frame._model.predict(normalized)
		
		
		# Add 4 columns of 0 onto predictions so it can be fed back through sc
		shaped_predictions = np.empty(shape = (1, 5))
		for row in range(0, 1):
			shaped_predictions[row, 0] = predicted_price[row, 0]
		for col in range (1, 5):
			shaped_predictions[row, col] = 0
		
		
		# undo normalization
		predicted_price = sc.inverse_transform(shaped_predictions)
		return predicted_price[0][0]
	

	def setup(NUMBARS, model, api):
		Time_frame._NUMBARS = NUMBARS
		Time_frame._model = model
		Time_frame._api = api
	
	def get_max_gain(_model):
		prediction_list = []
		frame_names = []
		for frame in Time_frame._frames:
			frame.get_prediction(_model)
			prediction_list.append(frame.prediction)
			frame_names.append(frame.frame_name)
		
		current_price = self.get_current_price()
		
		# predict percent gain
		highest_prediction = prediction_list.index(max(prediction_list))
		gain = prediction_list[highest_prediction]/current_price
		gain = round((gain -1) * 100, 3)
		
		gain_time_period = frame_names[highest_prediction]
		return gain, gain_time_period
