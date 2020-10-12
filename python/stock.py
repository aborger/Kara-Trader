from python.time_frame import Time_frame
BACKTEST = 'data/backTest/'

class Stock:
	_api = 0
	_period = 0
	_loss_percent = .01
	
	def setup(NUMBARS, model, api, period):
		Stock._stocks = []
		Time_frame.setup(NUMBARS, model, api)
		Stock._api = api
		Stock._period = Stock._convert_frame_name(period)
		
	def get_current_price(self):
		return self.frames[Stock._period].get_current_price()
		
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
	def collect_stocks(num_stocks):
		best_stocks = Stock._highest_gain(num_stocks)
		gain_sum = 0
		for stock in best_stocks:
			gain_sum += stock.frames[Stock._period].gain
		value_per_gain = 100/gain_sum
		stocks = []
		for stock in best_stocks:
			this_buy_ratio = stock.frames[Stock._period].gain * value_per_gain
			this_stock = dict(stock_object = stock, buy_ratio = this_buy_ratio/100)
			stocks.append(this_stock)
		return stocks
	# returns num_stocks best stocks
	def _highest_gain(num_stocks): 
				
		def get_gain(stock):
				return stock.frames[Stock._period].gain
		
		# Currently only using 5 max gains
		#print("Getting highest gain...")
		max_stocks = []
		for stock in Stock._stocks:
			try:
				stock.frames[Stock._period].get_gain()
			except ValueError:
				pass
			except:
				raise
			else:
				#print(stock.symbol + "'s gain is " + str(stock.frames[Stock._period].gain))
				if len(max_stocks) < num_stocks:
					max_stocks.append(stock)
				elif stock.frames[Stock._period].gain > max_stocks[num_stocks - 1].frames[Stock._period].gain:
					max_stocks.pop()
					max_stocks.append(stock)
				
				# sort list so lowest gain is at the end
				max_stocks.sort(reverse=True, key=get_gain)
		return max_stocks
	
	
	def __init__(self, symbol):
		self.symbol = symbol
		self._stocks.append(self)
		
		self.frames = [Time_frame('1Min', symbol), Time_frame('5Min', symbol),
						Time_frame('15Min', symbol), Time_frame('1D', symbol)]
		
	def get_gain(self):
		return self.frames[Stock._period].gain
	
	def buy(self, api, quantity):
		bought_price = self.frames[0].get_current_price()

		#self.stop_price = bought_price - (bought_price * Stock._loss_percent)
		#print ('Bought ' + str(quantity) + ' shares of ' + self.symbol
		#		+ ' at ' + str(bought_price) + '. Gain: ' + str(self.frames[Stock._period].gain))

		print ('Bought ' + self.symbol + ' QTY: ' + str(quantity))
		api.submit_order(
			symbol=self.symbol,
			qty=quantity,
			side='buy',
			type='market',
			time_in_force='gtc')


		
	def trailing_stop(name, api, quantity):
		print('Applying trailing stop: ')
		print(name)
		# submits trailing stop order
		api.submit_order(
			symbol=name,
			qty=quantity,
			side='sell',
			type='trailing_stop',
			time_in_force='gtc',
			trail_percent=1)
			
	def sell(self, api, quantity):
		#print('=====================================')
		#print ('Sold ' + self.symbol)
		api.submit_order(
			symbol=self.symbol,
			qty=quantity,
			side='sell',
			type='market',
			time_in_force='gtc')
			
	def sell_named_stock(name, api, quantity):
		#print('=====================================')
		print ('Sold ' + name + ' qty: ' + str(quantity))
		api.submit_order(
			symbol=name,
			qty=quantity,
			side='sell',
			type='market',
			time_in_force='gtc')
		#except:
		#	print('Cannot sell due to day trade restrictions')
		#finally:
			#pass


		



	
