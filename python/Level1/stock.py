from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from sklearn.preprocessing import MinMaxScaler
BACKTEST = 'data/backTest/'

class Stock:
	_NUMBARS = None
	_model = None
	_time_frame = None
	_loss_percent = .01
	_stocks = []

	#-----------------------------------------------------------------------#
	#								Initializing							#
	#-----------------------------------------------------------------------#
	def __init__(self, ticker):
		self.gain = None
		if isinstance(ticker, str): # its a regular string
			self.symbol = ticker
			Stock._stocks.append(self)
		else: # create a stock object from position object
			self.symbol = ticker.symbol


	@classmethod
	def setup(cls, NUMBARS, model, time_frame):
		cls._NUMBARS = NUMBARS
		cls._model = model
		cls._time_frame = time_frame
		


	#-----------------------------------------------------------------------#
	#								Individual								#
	#-----------------------------------------------------------------------#
		
	def find_current_price(self, api):
		barset = (api.get_barset(self.symbol,'1Min',limit=1))
		symbol_bars = barset[self.symbol]
		current_price = symbol_bars[0].c
		return current_price

	def find_prediction(self, api):
		# Get bars
		barset = (api.get_barset(self.symbol, Stock._time_frame, limit=Stock._NUMBARS))
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
		reshapedSet = np.reshape(npDataSet, (1, Stock._NUMBARS, 5))
		
		# Normalize Data
		sc = MinMaxScaler(feature_range=(0,1))
		normalized = np.empty(shape=(1, Stock._NUMBARS, 5)) 
		normalized[0] = sc.fit_transform(reshapedSet[0])
		
		# Predict Price
		predicted_price = Stock._model.predict(normalized)
		
		
		# Add 4 columns of 0 onto predictions so it can be fed back through sc
		shaped_predictions = np.empty(shape = (1, 5))
		for row in range(0, 1):
			shaped_predictions[row, 0] = predicted_price[row, 0]
		for col in range (1, 5):
			shaped_predictions[row, col] = 0
		
		
		# undo normalization
		predicted_price = sc.inverse_transform(shaped_predictions)
		return predicted_price[0][0]

	def find_gain(self, api):
			prediction = self.find_prediction(api)
			current = self.find_current_price(api)
			
			gain = prediction/current
			gain = round((gain -1) * 100, 3)
			if gain < 0:
				gain = 0
			self.gain = gain
			return gain

	
	#-----------------------------------------------------------------------#
	#									Trading								#
	#-----------------------------------------------------------------------#
	
	ACTUALLY_TRADE = False

	def buy(self, api, quantity):
		print ('Buying ' + self.symbol + ' QTY: ' + str(quantity))
		if ACTUALLY_TRADE:
			api.submit_order(
				symbol=self.symbol,
				qty=quantity,
				side='buy',
				type='market',
				time_in_force='gtc')
		else:
			print('WARNING, ACTUALLY TRADE = FALSE')

	def sell(self, api, quantity):
		print ('Sold ' + self.symbol)
		if ACTUALLY_TRADE:
			api.submit_order(
				symbol=self.symbol,
				qty=quantity,
				side='sell',
				type='market',
				time_in_force='gtc')
		else:
			print('WARNING, ACTUALLY TRADE = FALSE')
		
	def trailing_stop(self, api, quantity, percent):
		print('Applying trailing stop for ' + self.symbol)
		if ACTUALLY_TRADE:
			api.submit_order(
				symbol=self.symbol,
				qty=quantity,
				side='sell',
				type='trailing_stop',
				time_in_force='gtc',
				trail_percent=percent)
		else:
			print('WARNING, ACTUALLY TRADE = FALSE')
		
	#-----------------------------------------------------------------------#
	#								Calculations							#
	#-----------------------------------------------------------------------#
	
	# Main function used by tradeAI
	# Returns two items: diversified_stocks and second_best_stocks
	# diversified_stocks is dict with best stocks and their buy ratio
	# second_best_stocks is num_best_stocks next best stocks
	@classmethod
	def find_diversity(cls, num_best_stocks, api):
		best_stocks, all_best_stocks = Stock._find_best(num_best_stocks, api)
		gain_sum = 0
		for stock in best_stocks:
			gain_sum += stock.gain
		if gain_sum == 0:
			value_per_gain = 0
		else:
			value_per_gain = 100/gain_sum
		diversified_stocks = []
		for stock in best_stocks:
			this_buy_ratio = stock.gain * value_per_gain
			this_stock = dict(stock_object = stock, buy_ratio = this_buy_ratio/100)
			diversified_stocks.append(this_stock)
		return diversified_stocks, all_best_stocks
	
	@classmethod
	def _find_gain(cls, stock, api):
		stock.find_gain(api)
		return 0
		
	# returns tuple of two lists
	# list[0] = num_best_stocks of the highest gains. If num_best_stocks is 5, list[0] is the top 5 stocks
	# list[1] = next numb_best_stocks of the next highest gains. If num_best_stocks is 5, list[1] is the next top 5 stocks
	@classmethod
	def _find_best(cls, num_best_stocks, api): 
				
		def get_gain(stock):
				return stock.gain
		
		# find gain for every stock
		# use multiprocessing here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
		for stock in cls.get_stock_list():
			cls._find_gain(stock, api)



		# Add best gains to max_stocks
		max_stocks = []
		for stock in Stock._stocks:
				if len(max_stocks) < num_best_stocks * 2:
					max_stocks.append(stock)
				elif stock.gain > max_stocks[-1].gain:
					max_stocks.pop()
					max_stocks.append(stock)
				
		# sort list so lowest gain is at the end
		max_stocks.sort(reverse=True, key=get_gain)
		best = max_stocks[0:num_best_stocks]
		return best, max_stocks

	#-----------------------------------------------------------------------#
	#									Getters								#
	#-----------------------------------------------------------------------#

	@classmethod
	def get_stock_list(cls):
		return cls._stocks



		



	
