import alpaca_trade_api as tradeapi
from ...user import User as alpacaUser
import datetime
import pandas as pd
STOCKDIR = '../Stock_Data/'
#448 for 2019, 
#700 for 2018 (10/12/2020)
BARS_TO_COLLECT = 700
NUMBARS = 10

class api:
	alpacaUser.update_users(is_paper=True, tradeapi=tradeapi)
	_alpacaAPI = alpacaUser.get_api()
	def __init__(self, value):
		self.clock = Clock()
		self.account = Account(value)
		
	def _get_current(self, symbol):
		barset = self.get_barset(symbol, '1Min', 1)
		bars = barset[symbol]
		price = bars[0].c
		return price
		
	def get_account(self):
		return self.account
	
	def get_clock(self):
		return self.clock
	def get_barset(self, symbol, timeframe, limit):

		log = pd.read_csv(STOCKDIR + symbol + '.csv', sep=r'\s*,\s*', engine='python')
		log = log.to_numpy()
		new_barset = []
		# days_past starts at 10 meaning first day it will start at row 10 providing space to predict
		for row in range(self.clock.bars_past - limit, self.clock.bars_past):
			o = log[row][0]
			c = log[row][1]
			h = log[row][2]
			l = log[row][3]
			v = log[row][4]
			#new_bar = Bar(log[row][0], log[row][1], log[row][2], log[row][3], log[row][4])
			new_bar = Bar(o, c, h, l, v)

			new_barset.append(new_bar)
		barset = {
			symbol: new_barset
		}
		return barset
		
	def list_positions(self):
		return self.account.portfolio
		
	def reset(self):
		self.clock.reset_bars()
		self.account.reset()
		
	def next_bar(self):
		self.clock.next_bar()
		#print(self.account.equity)
		new_equity = 0
		for position in self.account.portfolio:
			price = self._get_current(position.symbol)
			position.update_price(price)
			new_equity += price * position.qty
		self.account.last_equity = self.account.equity
		self.account.equity = float(new_equity + self.account.buying_power)
		#print(self.account.equity)
			
	def submit_order(self, symbol, qty, side, type, time_in_force):
		# Get price
		price = self._get_current(symbol)
		
		new_position = Position(symbol, qty, price)
		
		if side == 'buy':
			return self.account.add_position(new_position)
		elif side == 'sell':
			return self.account.remove_position(new_position)
		else:
			print('Not an option')
		
	def get_data(stocks, timeframe):
		working_stocks = []
		for stock in stocks:
			print('Getting data for ' + stock)
			enough_data = False
			try:
				size = pd.read_csv(STOCKDIR + stock + '.csv', sep=r'\s*,\s*', engine='python').size
				if size == 5*(BARS_TO_COLLECT + NUMBARS):
					print('Already have data')
					working_stocks.append(stock)
					enough_data =  True
				else:
					print(size)
			except FileNotFoundError:
				pass
			except:
				raise

			if not enough_data:
				barset = api._alpacaAPI.get_barset(stock, timeframe, BARS_TO_COLLECT + NUMBARS)

				symbol_bars = barset[stock]
				if len(symbol_bars) != BARS_TO_COLLECT + NUMBARS:
					print('This stock doesnt have enough data')
				else:
					# Always rewrite because number of bars changes
					# Start log
					log = open(STOCKDIR + stock + '.csv','w')
					log.write('Open, Close, High, Low, Volume\n')
					
					
					for barNum in range(0, len(symbol_bars)):
						log.write(str(symbol_bars[barNum].o) + ',' + str(symbol_bars[barNum].c) + ',' + str(symbol_bars[barNum].h) + ',' + str(symbol_bars[barNum].l) + ',' + str(symbol_bars[barNum].v) + '\n')
					
					log.close()

					working_stocks.append(stock)
		return working_stocks

class Clock:
	def __init__(self):
		self.is_open = True
		#self.real_time = datetime.datetime.today()
		self.bars_past = NUMBARS # usually 10
		# so days_past will be at 1/2/19
		self.timestamp = BARS_TO_COLLECT + self.bars_past #448 for 2019, 700 for 2018
		# dont forget to change in get_data too
	'''	
	def set_time(self, day, month, year, hour, minute, second):
		self.timestamp = datetime.datetime(year, month, day, hour, minute, second)

		self.timestamp = self.real_time - self.timestamp

		#self.timestamp = self.timestamp.days
	'''	

	def reset_bars(self):
		self.bars_past = NUMBARS
	def next_bar(self):
		self.bars_past += 1
		
		
		
	
class Account:
	def __init__(self, value):
		self.portfolio = []
		self.equity = value
		self.last_equity = value
		self.buying_power = value
		self.status = 'Active'
		
	def add_position(self, position):
		# Add position to portfolio
		exists = False
		
		buying_value = position.qty * position.entry_price
		if buying_value > self.buying_power:
			#print('Not enough buying power')
			return False
		else:
			for positions in self.portfolio:
				if positions.symbol == position.symbol:
					positions.qty += position.qty
					exists = True
			if not exists:
				self.portfolio.append(position)
				
		
			self.buying_power -= buying_value
			return True
		
		
	def get_portfolio(self):
		return self.portfolio

	def reset(self):
		self.equity = 1000
		self.last_equity = 1000
		self.buying_power = 1000
		self.portfolio = []
	def remove_position(self, position):
		exists = False
		value = position.qty * position.entry_price
		for positions in self.portfolio:
			if positions.symbol == position.symbol:
				if positions.qty > position.qty:
					positions.qty -= position.qty
					exists = True
					self.buying_power += value
				
		
		#print('Value of ' + str(value))
				
		if not exists:
			#print('Not enough positions available to sell!')
			return False
		else:
			return True

	def remove_empty(self):
		remove_list = []
		for position in self.portfolio:
			if position.qty == 0:
				remove_list.append(position)
				
		for position in remove_list:
			self.portfolio.remove(position)
			
	def print_portfolio(self):
		for position in self.portfolio:
			print('         Back          ')
			print('-----------------------')
			print('Symbol: ' + position.symbol)
			print('Qty: ' + str(position.qty))
			print('Price: ' + str(position.current_price))
			print('Value: ' + str(position.qty * position.entry_price))
		
class Position:
	def __init__(self, symbol, qty, entry_price):
		self.symbol = symbol
		self.qty = qty
		self.entry_price = entry_price
		self.current_price = entry_price
		
	def update_price(self, new_price):
		self.current_price = new_price

class Bar:
	def __init__(self, o, c, h, l, v):
		self.o = o
		self.c = c
		self.h = h
		self.l = l
		self.v = v

def REST(key_id, secret_key, base_url):
		new_api = api(secret_key)
		return new_api
		
class rest:
	class APIError(Exception): 
		print('APIError has occured')
		
