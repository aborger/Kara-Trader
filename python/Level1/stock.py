import pathos
import math
import numpy as np

BACKTEST = 'data/backTest/'
INDICATOR_DATA_FILE = 'data/indicator_data.csv'
STOCK_DATA_DIR = 'data/stock_history/'
USE_MULTIPROCESSING = False
USE_GPU = True

MAX_NUM_STOCKS = 200 # Max number of stocks to call from alpaca api

class Stock():
	_NUMBARS = None
	_time_frame = None
	_loss_percent = .01
	_stocks = []
	_main_api = None
	ACTUALLY_TRADE = None

	#-----------------------------------------------------------------------#
	#								Initializing							#
	#-----------------------------------------------------------------------#
	def __init__(self, ticker):
		self.gain = None
		self.real_gain = None
		self.predicted_price = None
		self.prev_bars = None
		self.current_price = None
		if isinstance(ticker, str): # its a regular string
			self.symbol = ticker
			Stock._stocks.append(self)
		else: # create a stock object from position object
			self.symbol = ticker.symbol

	def __str__(self):
		#return 'Symbol: ' + self.symbol + ' Price: ' + str(self.current_price) + ' Prediction: ' + str(self.predicted_price) + ' Gain: ' + str(self.gain) + ' RGain: ' + str(self.real_gain)
		return 'Symbol: ' + self.symbol + ' num bars: ' + str(len(self.prev_bars)) + ' current price: ' + str(self.current_price)

	@classmethod
	def setup(cls, NUMBARS, model, time_frame, main_api, boosters, boost_users, is_test):
		cls._NUMBARS = NUMBARS
		cls._model = model
		cls._time_frame = time_frame
		cls._main_api = main_api
		cls._boosters = boosters
		cls._boost_users = boost_users
		cls.ACTUALLY_TRADE = not is_test

		if cls.ACTUALLY_TRADE:
			print('NOTE: YOU WILL BE ACTUALLY TRADING')
		else:
			print('NOTE: THIS IS A TEST, YOU ARE NOT ACTUALLY TRADING')

	@classmethod
	def unload_stocks(cls):
		stocks = cls._stocks
		cls._stocks = []
		return stocks

	@classmethod
	def _collect(cls, tickers, time_frame, limit):
		print('collecting...')
		# calculate number of workers
		print('num stocks: ' + str(len(tickers)))
		num_apis = len(cls._boosters)
		print('num apis: ' + str(num_apis))
		num_stocks_per_api = int(len(tickers) / num_apis)
		print('stocks per api: ' + str(num_stocks_per_api))
		if num_stocks_per_api > MAX_NUM_STOCKS:
			num_stocks_per_api = MAX_NUM_STOCKS
		print('stocks per api: ' + str(num_stocks_per_api))
		num_stocks_per_rep = num_apis * num_stocks_per_api
		print('stocks per rep: ' + str(num_stocks_per_rep))
		num_repitions = math.ceil(len(tickers) / num_stocks_per_rep)
		print('num repititions: ' + str(num_repitions))


		dataSet = []
		for repition in range(0, num_repitions):
			workers = []
			for api_num in range(0, num_apis):
				min_stock = repition * num_stocks_per_rep + api_num * num_stocks_per_api
				max_stock = repition * num_stocks_per_rep + (api_num + 1) * num_stocks_per_api
				if max_stock > len(tickers):
					max_stock = len(tickers)
				print('Repition: ' + str(repition) + ' api num: ' + str(api_num) + ' min stock: ' + str(min_stock) + ' max stock: ' + str(max_stock))

				worker_tickers = [stock.symbol for stock in cls._stocks[min_stock : max_stock]]
				worker_api = cls._boosters[api_num]
				worker_dict = dict(api = worker_api, tickers = worker_tickers)
				workers.append(worker_dict)

			pool = pathos.helpers.mp.Pool(num_apis)
			worker_users = []
			barset = pool.starmap(cls._access_api, [(worker, time_frame, limit) for worker in workers])
			print('map complete.')
			pool.close()
			print('Pool is closed')

			for work_data in barset:
				for stock in work_data:
					dataSet.append(stock)


		return dataSet


	@classmethod
	def _access_api(cls, worker, time_frame, limit):
		print('accessing api...')
		tickers = worker["tickers"]
		api = worker["api"]


		barset = api.get_barset(tickers, time_frame, limit=limit)
		stockSet = []
		for stock_num in range(0, len(barset)):
			ticker = tickers[stock_num]
			bars = barset[ticker]

			stock_data = []
			for barNum in bars:
				bar = dict(o = barNum.o, c = barNum.c, h = barNum.h, l = barNum.l, v = barNum.v)
				stock_data.append(bar)
			stock = dict(ticker = ticker, barSet = stock_data)
			stockSet.append(stock)
			
		return stockSet

	@classmethod
	def collect_current_prices(cls):
		# alpaca api only allows a certain number of stocks per api call
		# several calls are necessary
		
		print('Collecting current prices...')
		tickers = [x.symbol for x in cls._stocks]

		stockSet = cls._collect(tickers, '1Min', limit=1)


		old_stocks = cls.unload_stocks()
		updated_stocks = []
		for stock in stockSet:
			if len(stock['barSet']) == 0:
				print(stock['ticker'] + ' has no bars')
			elif stock['barSet'][0]['c'] == 0:
				print(stock['ticker'] + ' has a price of 0.')
			else:
				try:
					for old_stock in old_stocks:
						if old_stock.symbol == stock['ticker']:
							old_stock.current_price = stock['barSet'][0]['c']
							updated_stocks.append(old_stock)
				except:
					print(this.symbol)
					#print(stock['barSet'])
					raise

		cls._stocks = updated_stocks


	@classmethod
	def collect_prices(cls, num_bars):
		# figure out how many times you have to call the api
		tickers = [x.symbol for x in cls._stocks]

		stockSet = cls._collect(tickers, cls._time_frame, limit=num_bars)

		old_stocks = cls.unload_stocks()
		updated_stocks = []
		for stock in stockSet:
			if len(stock['barSet']) == 0 or len(stock['barSet']) < num_bars:
				print(stock['ticker'] + ' doesnt have enough bars')
			else:
				try:
					bars = []
					for bar in stock['barSet']:
						barSet = []
						barSet.append(bar['o'])
						barSet.append(bar['c'])
						barSet.append(bar['h'])
						barSet.append(bar['l'])
						barSet.append(bar['v'])
						bars.append(barSet)
					
					if np.any(np.array(bars) < 1):
						print(stock.symbol + ' has wack data.')
					else:
						for old_stock in old_stocks:
							if old_stock.symbol == stock['ticker']:
								old_stock.prev_bars = bars
								updated_stocks.append(old_stock)
				except Exception as e:
					print(stock['ticker'])
					print(e)
					#print(stock['barSet'])
					

		cls._stocks = updated_stocks
				
	
		good_stocks = []
		for stock in cls._stocks:
			if np.any(np.array(stock.prev_bars) < 1):
				print(stock.symbol + ' has some wack data')
			elif len(stock.prev_bars) == num_bars:
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
		if Stock.ACTUALLY_TRADE:
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
		if Stock.ACTUALLY_TRADE:
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
		if Stock.ACTUALLY_TRADE:
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
		if Stock.ACTUALLY_TRADE:
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
	def find_diversity(cls, num_best_stocks):
		best_stocks = Stock._find_best(num_best_stocks)
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
	def _find_best(cls, num_best_stocks): 
				
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
			stocks = GPU_find_gain(cls, cls._model, cls._NUMBARS)
			print('Number of viable stocks: ' + str(len(Stock.get_stock_list())))
			for i in range(0, len(cls.get_stock_list())):
				stocks_with_gains.append(stocks[i])
		else:
			from python.Level1.Level2.predict import find_gain
			from tensorflow import keras

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



		



	
