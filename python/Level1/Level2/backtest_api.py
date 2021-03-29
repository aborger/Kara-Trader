import alpaca_trade_api as tradeapi
from ...user import User as alpacaUser
import numpy as np
import math
from datetime import datetime, date, time, timedelta
import rfc3339 # datetime format
import pandas as pd
NUMBARS = 10


#-----------------------------------------------------------------------#
#									API									#
#-----------------------------------------------------------------------#

class api:
	alpacaUser.update_users(is_paper=True, tradeapi=tradeapi)
	_alpacaAPI = alpacaUser.get_api()
	def __init__(self, value):
		self.clock = None
		self.account = Account(value)
		self.day_barset = None
		self.min_barset = None

	def setup(self, symbols, num_days):
		self.clock = Clock(num_days)
		self.day_barset = Barset(symbols, '1D', start=self.clock.get_start_day(), limit=num_days)
		self.min_barset = Barset(symbols, '1Min', start=self.clock.get_open(), end=self.clock.get_close(), limit=self.clock.min_in_day)

	def list_assets(self, status):
		return api._alpacaAPI.list_assets(status=status)
		
	def _get_current(self, symbol):
		barset = self.get_barset(symbol, '1Min', 1)
		bars = barset[symbol]
		price = bars[0].c
		return price
		
	def get_account(self):
		return self.account
	
	def get_clock(self):
		return self.clock

	def get_barset(self, symbols, timeframe, limit):
		print('timeframe: ' + str(timeframe))
		if timeframe == '1Day':
			return self.day_barset.get_barset(symbols, self.clock.get_day_num(), limit)
		elif timeframe == '1Min':
			return self.min_barset.get_barset(symbols, self.clock.get_min_num(), limit)
		
	def list_positions(self):
		return self.account.portfolio
		
	def next_day(self):
		self.clock.next_day()
		
		new_equity = 0
		for position in self.account.portfolio:
			price = self._get_current(position.symbol)
			position.update_price(price)
			new_equity += price * position.qty
		self.account.last_equity = self.account.equity
		self.account.equity = float(new_equity + self.account.buying_power)
			
	def submit_order(self, symbol, qty, side, type, time_in_force):
		# Get price
		price = self._get_current(symbol)
		
		new_position = Position(symbol, qty, price)
		
		if side == 'buy':
			self.account.add_position(new_position)
		elif side == 'sell':
			self.account.remove_position(new_position)
		else:
			print('Not an option')
		

	

#-----------------------------------------------------------------------#
#									Clock								#
#-----------------------------------------------------------------------#

class Clock:
	def __init__(self, numdays):
		self.is_open = True
		self.days_past = NUMBARS # usually 10
		self.open = time(9, 30)
		self.close = time(16, 00)
		delta = self._time_diff(self.close, self.open)
		self.min_in_day = int(delta.seconds / 60) # should be 450

		now = date.today()
		days_back = timedelta(numdays - self.days_past)

		self.start_day = now - days_back
		self.daystamp = self.start_day
		self.timestamp = time(9, 30)

	def _time_diff(self, exit, enter):
		return datetime.combine(date.today(), exit) - datetime.combine(date.today(), enter)

	def _date_diff(self, exit, enter):
		return datetime.combine(exit, time(0)) - datetime.combine(enter, time(0))

	def next_day(self):
		delta = datetime.timedelta(days=1)
		self.daystamp = self.daystamp + 1

	def next_min(self):
		delta = datetime.timedelta(min=1)
		self.timestamp = self.timestamp + 1

	def get_day_stamp(self):
		return rfc3339.rfc3339(self.daystamp)

	def get_time_stamp(self):
		time = datetime.combine(self.daystamp, self.timestamp)
		return rfc3339.rfc3339(time)

	def get_day_num(self):
		delta = self._day_diff(self.daystamp, self.start_day)
		return delta.days

	def get_min_num(self):
		delta = self._time_diff(self.timestamp, self.open)
		return int(delta.seconds / 60)

	def get_start_day(self):
		time = datetime.combine(self.start_day, self.open)
		return rfc3339.rfc3339(time)

	def get_open(self):
		time = datetime.combine(self.daystamp, self.open)
		return rfc3339.rfc3339(time)

	def get_close(self):
		time = datetime.combine(self.daystamp, self.close)
		return rfc3339.rfc3339(time)


	
		
		
#-----------------------------------------------------------------------#
#									Account								#
#-----------------------------------------------------------------------#
	
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
		for positions in self.portfolio:
			if positions.symbol == position.symbol:
				positions.qty += position.qty
				exists = True
		if not exists:
			self.portfolio.append(position)
			
		# Subtract buying_power
		self.buying_power -= position.qty * position.entry_price
		
	def remove_position(self, position):
		exists = False
		value = position.qty * position.entry_price
		for positions in self.portfolio:
			if positions.symbol == position.symbol:
				positions.qty -= position.qty
				exists = True
		
				
		self.buying_power += value
		#print('Value of ' + str(value))
				
		if not exists:
			print('PositionNotInPortfolio')

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

#-----------------------------------------------------------------------#
#								Barset									#
#-----------------------------------------------------------------------#
class Barset:
	def __init__(self, symbols, time_frame, start=None, end=None, limit=None):
		self.MAX_NUM_STOCKS = 200
		self.symbols = symbols # list of symbol strings
		self.time_frame = time_frame
		self.start = start
		self.end = end
		self.limit = limit
		self.stockSet = {}

		self._collect()

		 

	def get_barset(self, symbols, bar_num, limit):
		stockset = self.stockSet
		print('num_bars: ')
		print(len(stockset[symbols[0]]))
		for stock in symbols:
			stockset[stock] = stockset[stock][bar_num - limit : bar_num]
		print(len(stockset[symbols[0]]))
		return stockset




	def _collect(self):
		# figure out how many times you have to call the api
		num_repititions = math.ceil(len(self.symbols) / self.MAX_NUM_STOCKS)

		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# use parallel processing and several apis for this
		for i in range(0, num_repititions):
			# figure out which stocks to get each time
			stocks_to_get = self.symbols[i*self.MAX_NUM_STOCKS:(i+1)*self.MAX_NUM_STOCKS]
			# call the api
			print('calling api...')
			if self.end is not None:
				barset = api._alpacaAPI.get_barset(stocks_to_get, self.time_frame, start=self.start, end=self.end)
			else:
				barset = api._alpacaAPI.get_barset(stocks_to_get, self.time_frame, start=self.start)
			print('setting up...')
			# add each stocks bars to its attribute
			self.stockSet = barset
	

		

#-----------------------------------------------------------------------#
#								Position								#
#-----------------------------------------------------------------------#
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
		

#-----------------------------------------------------------------------#
#									rest								#
#-----------------------------------------------------------------------#
class rest:
	class APIError(Exception): 
		print('APIError has occured')
		
