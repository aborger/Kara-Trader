# This file manages the other files

# Number of bars to predict on
# Ex: If NUMBARS=4 use monday-thursday to predict friday 
NUMBARS = 10
TRAINBARLENGTH = 1000
TRAINSET = 'data/dataset.csv'
TESTSET = 'data/testSet.csv'
MODELS = 'data/models/'

def train():
	import python.training.head_class as hc
	hc.Training_Model.oversee(TRAINSET, TESTSET, MODELS, args.name)
	
	

	
def trade(is_test, time_period):
	
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
	model = keras.models.load_model('data/Trade-Model.h5', compile=False)
	
	# update users first to gain access to the api
	from python.user_data.user import User
	User.update_users()
	
	# setup stocks
	from python.stock import Stock
	Stock.setup(NUMBARS, model, User.get_api(), time_period)
	for symbol in sp:
		this_stock = Stock(symbol)
	

		
	#while True:
	if User.get_api().get_clock().is_open:
		# At open, get 5 best stocks and their buy ratio
		best_stocks = Stock.collect_stocks(5)
		User.update_users()
		# Sell any open positions
		User.users_sell()
		# Buy the best stocks
		User.users_buy(best_stocks)
	else:
		print('Stock market is not open today.')
	
	return	
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
						help="'train', 'trade', 'test'")
	# Test
	parser.add_argument("-t", action='store_true', required=False,
						help='Include -t if this is a shortened test')
						
	parser.add_argument("--name", help = "Name for new model when training")
						
	parser.add_argument("--time", default='1D',
						help = "Time period to buy and sell on")
						
	args = parser.parse_args()

	# Run based on arguments
	if args.command == 'train':
		train()

	elif args.command == 'test':
		test(what_stocks, current_stock=0)

	elif args.command == 'trade':
		trade(args.t, args.time)
		
	else:
		raise InputError("Command must be either 'train', 'run', or 'view'")

