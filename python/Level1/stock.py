from python.Level1.Level2.time_frame import Time_frame
from pathos.multiprocessing import ProcessingPool as Pool
BACKTEST = 'data/backTest/'

class Stock:
	_period = 0
	_loss_percent = .01
	
	def __init__(self, symbol, NUMBARS, model):
		self.symbol = symbol
		self._stocks.append(self)
		
		self.frames = [Time_frame('1Min', symbol, NUMBARS, model), Time_frame('5Min', symbol, NUMBARS, model),
						Time_frame('15Min', symbol, NUMBARS, model), Time_frame('1D', symbol, NUMBARS, model)]
						
	def setup(NUMBARS, model, period):
		Stock._stocks = []
		Time_frame.setup(NUMBARS, model)
		Stock._period = Stock._convert_frame_name(period)
		
	def get_current_price(self, api):
		return self.frames[Stock._period].get_current_price(api)
		
	def _convert_frame_name(time_frame):
		if time_frame == '1Min':
			time_frame = 0
		elif time_frame == '5Min':
			time_frame = 1
		elif time_frame == '15Min':
			time_frame = 2
		elif time_frame == '1D':
			time_frame = 3
		else:
			raise InputError('Incorrect time frame')
		return time_frame
		
	# Main function used by trade ai
	# gives dict with best stocks and their buy ratio
	def collect_stocks(num_stocks, api_list):
		best_stocks = Stock._highest_gain(num_stocks, api_list)
		gain_sum = 0
		for stock in best_stocks:
			gain_sum += stock.frames[Stock._period].gain
		if gain_sum == 0:
			value_per_gain = 0
		else:
			value_per_gain = 100/gain_sum
		stocks = []
		for stock in best_stocks:
			this_buy_ratio = stock.frames[Stock._period].gain * value_per_gain
			this_stock = dict(stock_object = stock, buy_ratio = this_buy_ratio/100)
			stocks.append(this_stock)
		return stocks
	
	@classmethod
	def _find_gain(cls, stock, api):
		print(stock.symbol)
		stock.frames[Stock._period].get_gain(api)
		print('done')
		return 0

		
	# returns num_stocks best stocks
	@classmethod
	def _highest_gain(cls, num_stocks, api_list): 
				
		def get_gain(stock):
				return stock.frames[Stock._period].gain
		
		print('Number of stocks: ' + str(len(Stock._stocks)))
		
			
		print('calculating...')
		# find gain for every stock
		pool = Pool()
		done = pool.map(Stock._find_gain, Stock._stocks, api_list)

		print(done)
		
		# Add best gains to max_stocks
		# Currently only using 5 max gains
		max_stocks = []
		for stock in Stock._stocks:
				if len(max_stocks) < num_stocks:
					max_stocks.append(stock)
				elif stock.frames[Stock._period].gain > max_stocks[num_stocks - 1].frames[Stock._period].gain:
					max_stocks.pop()
					max_stocks.append(stock)
				
		# sort list so lowest gain is at the end
		max_stocks.sort(reverse=True, key=get_gain)
		return max_stocks
	
	
	
		
	def get_gain(self):
		return self.frames[Stock._period].gain
	
	def buy(self, api, quantity):
		#bought_price = self.frames[0].get_current_price()

		#self.stop_price = bought_price - (bought_price * Stock._loss_percent)
		#print ('Bought ' + str(quantity) + ' shares of ' + self.symbol
		#		+ ' at ' + str(bought_price) + '. Gain: ' + str(self.frames[Stock._period].gain))

		print ('Bought ' + self.symbol + ' QTY: ' + str(quantity))
		'''
		try:
			api.submit_order(
				symbol=self.symbol,
				qty=quantity,
				side='buy',
				type='market',
				time_in_force='gtc')
		except:
			print('Failed to buy')
			pass
		'''

		
	def trailing_stop(name, api, quantity, percent):
		print('Applying trailing stop: ')
		print(name)
		# submits trailing stop order
		api.submit_order(
			symbol=name,
			qty=quantity,
			side='sell',
			type='trailing_stop',
			time_in_force='gtc',
			trail_percent=percent)
			
	def sell(self, api, quantity):
		#print('=====================================')
		#print ('Sold ' + self.symbol)
		'''
		api.submit_order(
			symbol=self.symbol,
			qty=quantity,
			side='sell',
			type='market',
			time_in_force='gtc')
		'''
			
	def sell_named_stock(name, api, quantity):
		#print('=====================================')
		print ('Sold ' + name + ' qty: ' + str(quantity))
		'''
		api.submit_order(
			symbol=name,
			qty=quantity,
			side='sell',
			type='market',
			time_in_force='gtc')
		'''
		#except:
		#	print('Cannot sell due to day trade restrictions')
		#finally:
			#pass


		



	
