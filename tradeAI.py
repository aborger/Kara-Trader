# This file manages the other files

# Number of bars to predict on
# Ex: If NUMBARS=4 use monday-thursday to predict friday 
NUMBARS = 10
TRAINBARLENGTH = 1000
TRAINSET = 'data/dataset.csv'
TESTSET = 'data/testSet.csv'
MODELS = 'data/models/'
LOGDIR = 'data/logs/'

def train():
	import python.training.head_class as hc
	hc.Training_Model.oversee(TRAINSET, TESTSET, MODELS, args.name)
	

	
def test(is_paper):
	import alpaca_trade_api as tradeapi
	from python.user_data.user import User
	User.update_users(is_paper)
	User.get_stats()
	
def log(is_paper):
	import alpaca_trade_api as tradeapi
	from python.user_data.user import User
	User.update_users(is_paper)
	
	User.log(LOGDIR)
	
def read(is_paper):
	import alpaca_trade_api as tradeapi
	from python.user_data.user import User
	User.update_users(is_paper)
	
	User.view(LOGDIR)
	
	

def trade(is_test, time_period, is_paper):
	
	def wait_until_open():
		difference = 1
		while difference > 0:
			market_open = User.get_api().get_clock().next_open
			now = pd.Timestamp.now('US/Mountain')

			market_open = market_open.tz_convert('US/Mountain')

			difference = market_open - now

			print('Waiting for 1 minute.')
			sleep(60)
		
	#from python.stock import Stock
	#from python.PYkeys import Keys
	import alpaca_trade_api as tradeapi
	import pandas as pd
	from time import sleep
	
	# Load S&P500
	print('Loading stock list...')
	table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	df = table[0]
	sp = df['Symbol']
	
	if is_test:
		sp = sp[0:10]

	print('Loading AI...')
	from tensorflow import keras
	model = keras.models.load_model('data/models/different_stocks.h5', compile=False)
	
	# update users first to gain access to the api
	from python.user_data.user import User
	User.update_users(is_paper)
	
	# setup stocks
	from python.stock import Stock
	Stock.setup(NUMBARS, model, User.get_api(), time_period)
	for symbol in sp:
		this_stock = Stock(symbol)
	

		

	if User.get_api().get_clock().is_open:
		# At open, get 5 best stocks and their buy ratio
		#best_stocks = Stock.collect_stocks(5)
		#User.update_users(is_paper)
		# Sell any open positions
		User.users_sell()
		# Buy the best stocks
		#User.users_buy(best_stocks)
		'''
		ready = False
		while not ready:
			ready = User.check_bought()
		print('Now ready')
		User.users_trailing()
		'''
	else:
		print('Stock market is not open today.')
		
	# Wait until close
	#while User.get_api().get_clock().is_open:
	#	print('Waiting until closed...')
	#	sleep(60)
	#		else:
			# while closed
	#		while not User.get_api().get_clock().is_open:
	#			print('Waiting until open...')
	#			sleep(60)
	
		


	
#############################################
# Command Line
#############################################

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Control Trading AI')
	parser.add_argument("command", metavar="<command>",
						help="'train', 'trade', 'test', log, read")
	# Test
	parser.add_argument("-t", action='store_true', required=False,
						help='Include -t if this is a shortened test')
						
	parser.add_argument("--name", help = "Name for new model when training")
						
	parser.add_argument("--time", default='1D',
						help = "Time period to buy and sell on")
	parser.add_argument("-p", action='store_true', required=False,
						help='When trading include -f to only trade paper account')
						
	args = parser.parse_args()

	# Run based on arguments
	if args.command == 'train':
		train()

	elif args.command == 'test':
		test(args.p)

	elif args.command == 'trade':
		trade(args.t, args.time, args.p)
	
	elif args.command == 'log':
		log(args.p)
		
	elif args.command == 'read':
		read(args.p)
	else:
		raise InputError("Command must be either 'train', 'run', or 'view'")

