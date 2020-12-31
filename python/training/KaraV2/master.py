from python.training.KaraV2.Level1.train_rnn import Main
from python.training.KaraV2.Level1.train_rnn import config
import os
import shutil
from python.training.KaraV2.Level1.v2trade import backtest

LOGDIR = 'data/v2training/AI.'

def train(Stock, User):
	print('USER FOR TRAINING: ' + User.get_user_list()[0].info['email'])
	stocks = Stock.get_list()
	# Create a policy and test
	print('Creating first champion...')
	champion = Policy(User, Stock)
	champion.save()
	champion.learn_more()
	champion.save()



	'''
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
			print('The reigning champion won with ' + str(contender.points))
			contender.delete()
		# Continue training champion
		print('Training winner more')
		champion.learn_more()
		champion.save()
	'''
			

class Policy:
	_policy_id = 0
	def __init__(self, User, Stock):
		self.id = Policy._policy_id
		Policy._policy_id += 1
		self.trainer = Policy_trainer(User, Stock)
		# Training
		self.best = self.trainer.train()
		self.Stock = Stock
		self.User = User
		print('Checking effectiveness...')
		# make save
		if os.path.exists(LOGDIR + str(self.id)):
			shutil.rmtree(LOGDIR + str(self.id))
		os.mkdir(LOGDIR + str(self.id))

		# Log stats
		log = open(LOGDIR + str(self.id) + '/stats.txt','w')
		log.write('Equity = ' + str(self.best["equity"]) + '\n' + 
		'Reward = ' + str(self.best["reward"]) + '\n' +
		'Actions = ' + str(self.best["actions"]) + '\n')
		log.close()

		# Validation
		for user in User.get_user_list():
			user.get_user_api().reset()
		self.points = backtest(config.NUMTRADES, self.best["model"], Stock, User, self.id)
		
		
	def learn_more(self):
		# Train model more
		v2Best = self.trainer.continue_training(self.best["model"])
		self.id += 0.1
		# if already there, remove
		if os.path.exists(LOGDIR + str(self.id)):
			shutil.rmtree(LOGDIR + str(self.id))
		# make new dir
		os.mkdir(LOGDIR + str(self.id))
		# See how it does
		for user in self.User.get_user_list():
			user.get_user_api().reset()
		v2points = backtest(NUMTRADES, v2Best["model"], self.Stock, self.User, self.id)
		# If it does better keep it
		if v2points > self.points:
			self.best["model"].set_weights(v2best["model"].get_weights())
			self.points = v2points
		
	def save(self):
		self.best["model"].save("data/v2training/AI." + str(self.id) + "/model")
	def delete(self):
		shutil.rmtree("data/v2training/AI." + str(self.id))
		
class Policy_trainer:
	def __init__(self, User, Stock):
		self.User = User
		self.Stock = Stock
		# Give Q functions
		# act = possible move
		self.Q = Main(self.act_buy, self.act_sell, self.act_wait, self.observe, self.reward, self.reset)
		
	# --------------------- Actions --------------------
	def act_buy(self, stock_num):
		stock = self.Stock.get_list()[stock_num]
		success = stock.buy(self.User.get_api(), 1)
		self.User.next_bar()
		return success

	def act_sell(self, stock_num):
		stock = self.Stock.get_list()[stock_num]
		success = stock.sell(self.User.get_api(), 1)
		self.User.next_bar()
		return success

	def act_wait(self, stock_num):
		#print('Next Day')
		self.User.next_bar()
		return True

	# ----------------------------------------------------
	def observe(self):
		# total_state[0] = stock_bars[NUMSTOCKS, NUMBARS, 5]. total_state[1] = position_size[NUMSTOCKS]. total_state[2] = buying_power_ratio[NUMSTOCKS]
		stocks = self.Stock.get_list()

		buying_power = self.User.get_api().get_account().buying_power

		stock_bars = []
		position_size = []
		buying_power_ratios = []

		for stock in stocks:
			# stock bars
			stock_bars.append(stock.get_prev_bars())

			# position size
			position = self.User.get_api().get_position(stock.symbol)
			if position is None:
				position_size.append(0)
			else:
				position_size.append(position.qty)

			# buying power ratios
			buying_power_ratio = buying_power / stock.get_current_price()
			buying_power_ratios.append(buying_power_ratio)

		return (stock_bars, position_size, buying_power_ratios)

	def reward(self):
		stocks = self.Stock.get_list()
		gains = []
		for stock in stocks:
			position = self.User.get_api().get_position(stock.symbol)
			if position is None:
				gains.append(0)
			else:
				gain = (position.current_price - position.entry_price) / position.entry_price
				gains.append(gain)

		equity = self.User.get_api().get_account().equity
		
		return (gains, equity)

	def reset(self):
		self.User.reset()	

	def train(self):
		print('Training...')
		return self.Q.train()
		
	def continue_training(self, model):
		print('Training...')
		return self.Q.continue_training(model)
		
	

	