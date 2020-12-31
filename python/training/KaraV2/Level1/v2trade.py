import tensorflow as tf
import numpy as np
from python.training.KaraV2.Level1.train_rnn import feed_EDQN, format_CDQN_output

def backtest(numdays, model, Stock, User, id):
	for day in range(0, numdays):
		User.log_single("data/v2training/ai." + str(id) + "/")
		trade(model, Stock, User)
		for user in User.get_user_list():
			user.api.get_account().remove_empty()			
		User.next_bar()

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

			buying_power = user.get_user_api().get_account().buying_power

			stock_bars = []
			position_size = []
			buying_power_ratios = []

			for stock in stocks:
				# stock bars
				stock_bars.append(stock.get_prev_bars())

				# position size
				position = user.get_user_api().get_position(stock.symbol)
				if position is None:
					position_size.append(0)
				else:
					position_size.append(position.qty)

				# buying power ratios
				buying_power_ratio = buying_power / stock.get_current_price()
				buying_power_ratios.append(buying_power_ratio)

			total_state = (stock_bars, position_size, buying_power_ratios)
			mid_state = feed_EDQN(model[0], total_state)
			mid_output = model[1](mid_state)
			output = format_CDQN_output(output)



			action = Actions(output[1], user)
			action.perform(output[0])
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