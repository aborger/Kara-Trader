import pathos

BACKTEST = 'data/backTest/'
INDICATOR_DATA_FILE = 'data/indicator_data.csv'
STOCK_DATA_DIR = 'data/stock_history/'
USE_MULTIPROCESSING = False
ACTUALLY_TRADE = False

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
		self.current_price = None
		if isinstance(ticker, str): # its a regular string
			self.symbol = ticker
			Stock._stocks.append(self)
		else: # create a stock object from position object
			self.symbol = ticker.symbol


	@classmethod
	def setup(cls, NUMBARS, model, time_frame, main_api):
		cls._NUMBARS = NUMBARS
		cls._model = model
		cls._time_frame = time_frame
		cls._main_api = main_api

		


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
					time_in_force='gtc',
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
		best_stocks, all_best_stocks = Stock._find_best(num_best_stocks, boosters)
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

		# Add best gains to max_stocks
		max_stocks = []
		for stock in stocks_with_gains:
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



		



	
