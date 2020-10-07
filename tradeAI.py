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
	

	
def test():
	User.get_stats()
	
def log():
	User.log(LOGDIR)
	
def read():
	User.view(LOGDIR)
	
def quick_sell():

	# Sell any open positions
	User.users_sell()
	
def trailing(is_paper):
	User.users_trailing()

def trade(is_test, time_period):
		
	#from python.stock import Stock
	#from python.PYkeys import Keys
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
	
	
	# setup stocks
	from python.stock import Stock
	Stock.setup(NUMBARS, model, User.get_api(), time_period)
	for symbol in sp:
		this_stock = Stock(symbol)
	

		

	if User.get_api().get_clock().is_open:
		# At open, get 5 best stocks and their buy ratio
		best_stocks = Stock.collect_stocks(5)
		# Sell any open positions
		User.users_sell()
		# Buy the best stocks
		User.users_buy(best_stocks)
	else:
		print('Stock market is not open today.')
		
	
#############################################
# Command Line
#############################################

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Control Trading AI')
	parser.add_argument("command", metavar="<command>",
						help="'train', 'trade', 'sell', 'test', 'trail', 'log', 'read'")
	# Test
	parser.add_argument("-t", action='store_true', required=False,
						help='Include -t if this is a shortened test')
						
	parser.add_argument("--name", help = "Name for new model when training")
						
	parser.add_argument("--time", default='1D',
						help = "Time period to buy and sell on")
	parser.add_argument("-b", action='store_true', required=False,
						help='Include -b to backtest')
	parser.add_argument("-p", action='store_true', required=False,
						help='When trading include -f to only trade paper account')
			
	args = parser.parse_args()

	# Run based on arguments
	if args.b:
		import python.backtest_api as tradeapi
		LOGDIR = 'data/backTest/logs/'
		from python.user_data.user import backtestUser as User
		User.update_users(tradeapi)
	else:
		print('I screwed up')
		import alpaca_trade_api as tradeapi
		from python.user_data.user import User
		User.update_users(args.p, tradeapi)
	if args.command == 'train':
		train()

	elif args.command == 'test':
		test()

	elif args.command == 'trade':
		trade(args.t, args.time,)
		
	elif args.command == 'sell':
		quick_sell()
		
	elif args.command == 'trail':
		trailing()
	
	elif args.command == 'log':
		log()
		
	elif args.command == 'read':
		read()
	else:
		raise InputError("Command must be either 'train', 'run', or 'view'")

