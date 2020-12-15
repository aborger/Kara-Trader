from python.training.KaraV2.Level1.train_rnn import Main
import os
import shutil
import numpy as np
from python.training.KaraV2.Level1.v2trade import backtest

NUMDAYS = 10

def train(Stock, User):
	print('USER FOR TRAINING: ' + User.get_user_list()[0].info['email'])
	stocks = Stock.get_list()
	# Create a policy and test
	print('Creating first champion...')
	champion = Policy(stocks[0], User, Stock)
	for stock in stocks:

		# Create another policy and test
		print('Creating a contender!')
		contender = Policy(stock, User, Stock)
		# Keep best policy to keep fighting other policies
		if contender.points > champion.points:
			print('The contender won with ' + str(contender.points))
			champion.delete()
			champion = contender
		else:
			print('The champion won with ' + str(contender.points))
			contender.delete()
		# Continue training champion
		print('Teaching the champion more')
		champion.learn_more()
		champion.save()
			

class Policy:
	_policy_id = 0
	def __init__(self, stock, User, Stock):
		self.id = Policy._policy_id
		Policy._policy_id += 1
		self.trainer = Policy_trainer(stock, User)
		# Training
		self.model = self.trainer.train()
		self.Stock = Stock
		self.User = User
		print('Checking effectiveness...')
		# make save
		if os.path.exists('data/v2training/' + str(self.id)):
			shutil.rmtree('data/v2training/' + str(self.id))
		os.mkdir('data/v2training/' + str(self.id))
		# Validation
		self.points = backtest(NUMDAYS, self.model, Stock, User, self.id)
		
		
	def learn_more(self):
		# Train model more
		v2model = self.trainer.continue_training(self.model)
		if os.path.exists('data/v2training/' + str(self.id + 100)):
			shutil.rmtree('data/v2training/' + str(self.id + 100))
		os.mkdir('data/v2training/' + str(self.id + 100))
		# See how it does
		v2points = backtest(NUMDAYS, v2model, self.Stock, self.User, self.id + 100)
		# If it does better keep it
		if v2points > self.points:
			self.model.set_weights(v2model.get_weights())
			self.points = v2points
		
	def save(self):
		self.model.save("data/v2training/" + str(self.id) + "/model")
	def delete(self):
		shutil.rmtree("data/v2training/" + str(self.id))
		
class Policy_trainer:
	def __init__(self, stock, User):
		self.stock = stock
		self.User = User
		# Give Q functions
		# act = possible move
		self.Q = Main(self.act_buy, self.act_sell, self.act_wait, self.observe, self.reward, self.reset)
		
	# --------------------- Actions --------------------
	def act_buy(self):
		success = self.stock.buy(self.User.get_api(), 1)
		self.User.next_day()
		return success

	def act_sell(self):
		success = self.stock.sell(self.User.get_api(), 1)
		self.User.next_day()
		return success

	def act_wait(self):
		#print('Next Day')
		self.User.next_day()
		return True

	# ----------------------------------------------------
	def observe(self):
		positions = self.User.get_api().list_positions()
		position_qty = 0
		for pos in positions:
			if pos.symbol == self.stock.symbol:
				position_qty = pos.qty

		dataSet = [self.User.get_api().get_account().buying_power, position_qty]
		npDataSet = np.array(dataSet)
		reshapedSet = np.reshape(npDataSet, (1, 2))

		return (self.stock.get_prev_bars(), reshapedSet)

	def reward(self):
		equity = self.User.get_api().get_account().equity
		#print('Equity = ' + str(equity))
		return equity
	def reset(self):
		self.User.reset()	

	def train(self):
		print('Training...')
		return self.Q.train()
		
	def continue_training(self, model):
		print('Training...')
		return self.Q.continue_training(model)
		
	

	