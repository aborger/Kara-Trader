import pathos
import math

BACKTEST = 'data/backTest/'
INDICATOR_DATA_FILE = 'data/indicator_data.csv'
STOCK_DATA_DIR = 'data/stock_history/'
USE_MULTIPROCESSING = False
USE_GPU = True
ACTUALLY_TRADE = False

MAX_NUM_STOCKS = 200 # Max number of stocks to call from alpaca api

class Stock():
	_NUMBARS = None
	_time_frame = None
	_loss_percent = .01
	_stocks = []
	_main_api = None

	#-----------------------------------------------------------------------#
	#								Initializing							#
	#-----------------------------------------------------------------------#
	def __init__(self, ticker):
		self.gain = None
		self.real_gain = None
		self.predicted_price = None
		self.prev_bars = None
		self.current_price = None
		self.trail_price = 0
		self.stop_price = 0
		if isinstance(ticker, str): # its a regular string
			self.symbol = ticker
			Stock._stocks.append(self)
		else: # create a stock object from position object
			self.symbol = ticker.symbol
			Stock._stocks.append(self)

	def __str__(self):
		return 'Symbol: ' + self.symbol + ' Price: ' + str(self.current_price) + ' Prediction: ' + str(self.predicted_price) + ' Gain: ' + str(self.gain) + ' RGain: ' + str(self.real_gain)


	@classmethod
	def setup(cls, NUMBARS, model, time_frame, main_api):
		cls._NUMBARS = NUMBARS
		cls._model = model
		cls._time_frame = time_frame
		cls._main_api = main_api

	@classmethod
	def unload_stocks(cls):
		cls._stocks = []

	@classmethod
	def set_stocks(cls, stock_symbols):
		cls.unload_stocks()
		for stock in stock_symbols:
			Stock(stock)

	@classmethod
	def collect_current_prices(cls):
		# alpaca api only allows a certain number of stocks per api call
		# several calls are necessary
		stock_symbols = [x.symbol for x in cls._stocks]
		num_repititions = math.ceil(len(stock_symbols) / MAX_NUM_STOCKS)
		
		# get prices and add to stock attribute
		for i in range(0, num_repititions):
			stocks_to_get = stock_symbols[i*MAX_NUM_STOCKS:(i+1)*MAX_NUM_STOCKS]
			barset = cls._main_api.get_barset(stocks_to_get, 'minute', limit=1)
			for stock_num in range(0, len(barset)):
				symbol = stocks_to_get[stock_num]
				this_stocks_bars = barset[symbol]
				price = -1
				try:
					price = this_stocks_bars[0].c
				except:
					print(cls._stocks[i*MAX_NUM_STOCKS + stock_num].symbol + 'does not have price')
				cls._stocks[i*MAX_NUM_STOCKS + stock_num].current_price = price

		"""
		good_stocks = []
		for stock in cls._stocks:
			if stock.current_price == None:
				good_stocks.append(stock)
				print(stock)
		#cls._stocks = good_stocks
		"""
	@classmethod
	def collect_prices(cls, time_frame, num_bars):
		# figure out how many times you have to call the api
		stock_symbols = [x.symbol for x in cls._stocks]
		num_repititions = math.ceil(len(stock_symbols) / MAX_NUM_STOCKS)

		for i in range(0, num_repititions):
			# figure out which stocks to get each time
			stocks_to_get = stock_symbols[i*MAX_NUM_STOCKS:(i+1)*MAX_NUM_STOCKS]
			# call the api
			barset = cls._main_api.get_barset(stocks_to_get, time_frame, limit=num_bars)
			# add each stocks bars to its attribute
			for stock_num in range(0, len(barset)):
				symbol = stocks_to_get[stock_num]
				this_stocks_bars = barset[symbol]

				dataSet = []
				for barNum in this_stocks_bars:
					bar = []
					bar.append(barNum.o)
					bar.append(barNum.c)
					bar.append(barNum.h)
					bar.append(barNum.l)
					bar.append(barNum.v)
					dataSet.append(bar)

				cls._stocks[i*MAX_NUM_STOCKS + stock_num].prev_bars = dataSet

		good_stocks = []
		for stock in cls._stocks:
			if len(stock.prev_bars) == num_bars:
				good_stocks.append(stock)

		cls._stocks = good_stocks
		
		


	#-----------------------------------------------------------------------#
	#								Individual								#
	#-----------------------------------------------------------------------#
		
	def find_current_price(self):
		barset = (Stock._main_api.get_barset(self.symbol,'1Min',limit=1))
		symbol_bars = barset[self.symbol]
		current_price = symbol_bars[0].c
		return current_price

	
	def set_stats(self, gain, real_gain, predicted, current):
		self.gain = gain
		self.real_gain = real_gain
		self.predicted_price = predicted
		self.current_price = current


	
	#-----------------------------------------------------------------------#
	#									Trading								#
	#-----------------------------------------------------------------------#
	def buy(self, api, quantity):
		print ('Buying ' + self.symbol + ' QTY: ' + str(quantity))
		if ACTUALLY_TRADE:
			try:
				api.submit_order(
					symbol=self.symbol,
					qty=quantity,
					side='buy',
					type='market',
					time_in_force='gtc')
			except Exception as exc:
				print(exc)
		else:
			print('WARNING, ACTUALLY TRADE = FALSE')

	def buy_notional(self, api, dollar_amount):
		print('Buying ' + self.symbol + ' Dollar Amount: ' + str(dollar_amount))
		if ACTUALLY_TRADE:
			data = {
				"symbol": self.symbol,
				"notional": dollar_amount,
				"side": 'buy',
				"type": 'market',
				"time_in_force": "day"
			}
			try:
				api._request('POST', '/orders', data=data)
			except Exception as exc:
				print(exc)
		else:
			print('WARNING, ACTUALLY TRADE = FALSE')

	def sell(self, api, quantity):
		print ('Sold ' + self.symbol)
		if ACTUALLY_TRADE:
			try:
				api.submit_order(
					symbol=self.symbol,
					qty=quantity,
					side='sell',
					type='market',
					time_in_force='day')
			except Exception as exc:
				print(exc)
		else:
			print('WARNING, ACTUALLY TRADE = FALSE')
		
	def trailing_stop(self, api, quantity, percent):
		print('Applying trailing stop for ' + self.symbol)
		if ACTUALLY_TRADE:
			try:
				api.submit_order(
					symbol=self.symbol,
					qty=quantity,
					side='sell',
					type='trailing_stop',
					time_in_force='day',
					trail_percent=percent)
			except Exception as exc:
				print(exc)
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
	def find_diversity(cls, num_best_stocks, boosters):
		best_stocks = Stock._find_best(num_best_stocks, boosters)
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
		return diversified_stocks
	

		
	# returns tuple of two lists
	# list[0] = num_best_stocks of the highest gains. If num_best_stocks is 5, list[0] is the top 5 stocks
	# list[1] = next numb_best_stocks of the next highest gains. If num_best_stocks is 5, list[1] is the next top 5 stocks
	@classmethod
	def _find_best(cls, num_best_stocks, boosters): 
				
		def get_gain(stock):
				return stock.gain

		# find gain for every stock
		stocks_with_gains = []
		if USE_MULTIPROCESSING:
			from python.Level1.Level2.predict import find_gains
			# use multiprocessing to speed up
			
			# find number of workers
			num_workers = None
			if pathos.helpers.mp.cpu_count() < len(boosters):
				num_workers = pathos.helpers.mp.cpu_count()
				print('Get more CPUs!')
			else:
				num_workers = len(boosters)
				print('Get more boosters!')

			print('num workers: ' + str(num_workers))

			# divide stocks per worker
			stocks_per_worker = int(len(cls._stocks) / num_workers)
			left_over = len(cls._stocks) % num_workers

			workers = []
			for worker in range(0, num_workers):
				min_stock = worker * stocks_per_worker
				max_stock = (worker + 1) * stocks_per_worker
				if worker == num_workers - 1:
					max_stock += left_over
				worker_stocks = cls._stocks[min_stock : max_stock]
				worker_api = boosters[worker]
				worker_dict = dict(api = worker_api, stocks = worker_stocks)
				workers.append(worker_dict)

			


			pool = pathos.helpers.mp.Pool(num_workers)
			predicted_stocks = pool.starmap(find_gains, [(worker, cls._time_frame, cls._NUMBARS) for worker in workers])
			pool.close()
		
			for group in predicted_stocks:
				stocks_with_gains = stocks_with_gains + group
		elif USE_GPU:
			from python.Level1.Level2.predict import GPU_find_gain
			stocks = GPU_find_gain(cls, cls._model, cls._time_frame, cls._NUMBARS)

			for i in range(0, len(cls.get_stock_list())):
				stocks_with_gains.append(stocks[i])
		else:
			from python.Level1.Level2.predict import find_gain
			from tensorflow import keras

			#model = keras.models.load_model('data/models/different_stocks.h5', compile=False)
			for stock in cls.get_stock_list():
				try:
					stocks_with_gains.append(find_gain(stock, cls._main_api, cls._model, cls._time_frame, cls._NUMBARS))
				except Exception as exc:
					print(exc)

		# Prepare to record data
		log = open(INDICATOR_DATA_FILE, 'w')	
		log.write('Stock, Predicted Gain, Predicted Price, Price\n')
		log.close()

		# Record data
		log = open(INDICATOR_DATA_FILE, 'a')
		for stock in stocks_with_gains:	
			log.write(stock.symbol + ', ' + str(stock.real_gain) + ', ' + str(stock.predicted_price) + ', ' + str(stock.current_price) + '\n')
		log.close()
				
		# sort list so lowest gain is at the end
		stocks_with_gains.sort(reverse=True, key=get_gain)
	
		best = stocks_with_gains[:num_best_stocks]
		return best


	#-----------------------------------------------------------------------#
	#									Logging								#
	#-----------------------------------------------------------------------#

	@classmethod
	def log_bars(cls, symbol, num_bars):
			barset = (Stock._main_api.get_barset(symbol, Stock._time_frame, limit=num_bars))
			symbol_bars = barset[symbol]

			# Prepare to record data
			log = open(STOCK_DATA_DIR + symbol + '.csv', 'w')	
			log.write('Timestamp, Value\n')
			
			# Record data
			for bar in range(0, num_bars):
				log.write(str(symbol_bars[bar].t) + ', ' + str(symbol_bars[bar].c) + '\n')

			log.close()



	#-----------------------------------------------------------------------#
	#									Getters								#
	#-----------------------------------------------------------------------#

	@classmethod
	def get_stock_list(cls):
		return cls._stocks



		



	
