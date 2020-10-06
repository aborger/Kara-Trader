import alpaca_trade_api as tradeapi
from python.user_data.user import User
import datetime

class api:
	User.update_users(is_paper=True, tradeapi=tradeapi)
	_alpacaAPI = User.get_api()
	def __init__(self):
		self.clock = Clock()
		self.account = Account()
		
	def get_account(self):
		return self.account
	
	def get_clock(self):
		return self.clock
	
	def get_barset(self, symbol, timeframe, limit):
		barset = api._alpacaAPI.get_barset(symbol, timeframe, limit=limit, end=self.clock.timestamp)
		return barset
		
	def list_positions(self):
		return self.account.portfolio
		
	def submit_order(self, symbol, qty, side, type, time_in_force):
		# Get price
		barset = self.get_barset(symbol, '1Min', 1)
		bars = barset[symbol]
		price = bars[0].c
		
		new_position = Position(symbol, qty, price)
		
		if side == 'buy':
			self.account.add_position(new_position)
		elif side == 'sell':
			self.account.remove_position(new_position)
		else:
			print('Not an option')
		

		
class Clock:
	def __init__(self):
		self.is_open = True
		self.real_time = datetime.datetime.now(datetime.timezone.utc)

		self.timestamp = self.real_time
		#self.timestamp = self.timestamp.replace(month=self.timestamp.month - 1)
		
	def set_time(self, day, month, year, hour=10, minute=0, second=0):
		self.timestamp = datetime.datetime(year, month, day, hour, minute, second)
		
	def next_day(self):
		self.timestamp = self.timestamp.replace(day=self.timestamp.day + 1)
		
		
	
class Account:
	def __init__(self):
		self.portfolio = []
		self.equity = 1000
		self.buying_power = 1000
		
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
		for positions in self.portfolio:
			if positions.symbol == position.symbol:
				positions.qty -= position.qty
				exists = True
				
		self.buying_power += position.qty * position.entry_price
				
		if not exists:
			print('PositionNotInPortfolio')
		print(self.buying_power)
		
class Position:
	def __init__(self, symbol, qty, entry_price):
		self.symbol = symbol
		self.qty = qty
		self.entry_price = entry_price

def REST(key_id, secret_key, base_url):
		new_api = api()
		return new_api
		
class rest:
	class APIError(Exception): 
		print('APIError has occured')
		pass
		