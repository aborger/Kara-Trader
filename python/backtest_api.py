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
		barset = api._alpacaAPI.get_barset(symbol, timeframe, limit, end=self.clock.real_time)
		return barset
		
	def list_positions(self):
		return self.account.positions
		
	def submit_order(self, symbol, qty, side, type, time_in_force):
		print('order of ' + symbol + ' complete')
		

		
class Clock:
	def __init__(self):
		self.is_open = True
		self.real_time = datetime.datetime.now(datetime.timezone.utc)
		print(self.real_time)
		self.timestamp = self.real_time
		self.timestamp.month -= 1
		
		
	
class Account:
	def __init__(self):
		self.positions = []
		self.buying_power = 1000
		
class Position:
	def __init__(self):
		self.symbol = 0
		self.qty = 0

def REST(key_id, secret_key, base_url):
		new_api = api()
		return new_api
		
class rest:
	class APIError(Exception): 
		print('APIError has occured')
		pass
		