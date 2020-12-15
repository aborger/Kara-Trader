import tensorflow as tf
import numpy as np

def backtest(numdays, model, Stock, User, id):
	for day in range(0, numdays):
		User.log("data/v2training/" + str(id) + "/")
		trade(model, Stock, User)
		for user in User.get_user_list():
			user.api.get_account().remove_empty()
			
		User.get_portfolio()
		User.next_day()
		print('                       Next Day')
		print('=======================================================')
		User.get_portfolio()
	equity_sum = 0
	for user in User.get_user_list():
		equity_sum += user.get_equity()
	average_equity = equity_sum / len(User.get_user_list())
	return average_equity
		
def trade(model, Stock, User):
		
	#from python.stock import Stock
	#from python.PYkeys import Keys
	from time import sleep



	if User.get_api().get_clock().is_open:
		for user in User.get_user_list():
			for stock in Stock.get_list():
				# get account data
				positions = user.get_user_api().list_positions()
				position_qty = 0
				for pos in positions:
					if pos.symbol == stock.symbol:
						position_qty = pos.qty

				dataSet = [user.get_user_api().get_account().buying_power, position_qty]
				npDataSet = np.array(dataSet)
				reshapedSet = np.reshape(npDataSet, (1, 2))

				state = (stock.get_prev_bars(), reshapedSet)
				action_num = tf.argmax(model(state)[0]).numpy()
				action = Actions(stock, user)
				action.perform(action_num)
	else:
		print('Stock market is not open today.')
		
		
		
class Actions:
	def __init__(self, stock, user):
		self.stock = stock
		self.user = user
		self.actions = [self.buy, self.sell, self.wait]
	def buy(self):
		self.stock.buy(self.user.get_user_api(), 1)
	def sell(self):
		self.stock.sell(self.user.get_user_api(), 1)
	def wait(self):
		#print('Next Day')
		self.user.user_next_day()
	def perform(self, action_num):
		action = self.actions[action_num]
		action()