# This file manages the other files

# Number of bars to predict on
# Ex: If NUMBARS=4 use monday-thursday to predict friday 

import pandas as pd

NUMBARS = 10
TRAINBARLENGTH = 1000
TRAINSET = 'data/dataset.csv'
TESTSET = 'data/testSet.csv'
MODELS = 'data/models/'
LOGDIR = 'data/logs/'
STOCKDIR = '../Stock_data/'


#===================================================================#
#								Run Functions						#
#===================================================================#
def train(name, Stock):
	#import python.training.head_class as hc
	#hc.Training_Model.oversee(TRAINSET, TESTSET, MODELS, name)
	from python.training.KaraV2.master import train
	print(type(User))
	train(Stock, User)
	
	
def backtest(numdays, model, Stock):
	from python.training.KaraV2.Level1.v2trade import trade as v2trade
	for day in range(0, numdays):
		log()
		v2trade(model, Stock, User)
		for user in User.get_user_list():
			user.api.get_account().remove_empty()
			
		User.get_portfolio()
		User.next_day()
		print('                       Next Day')
		print('=======================================================')
		User.get_portfolio()

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

def trade(model, Stock):
		
	#from python.stock import Stock
	#from python.PYkeys import Keys
	from time import sleep


	if True: #User.get_api().get_clock().is_open:
		User.users_sell()
		# At open, get 5 best stocks and their buy ratio
		print('Calculating best stocks...')
		best_stocks = Stock.collect_stocks(5)
		# Sell any open positions
		
		# Buy the best stocks
		User.users_buy(best_stocks)
	else:
		print('Stock market is not open today.')
		
#===================================================================#
#							Helping Functions						#
#===================================================================#
def import_data(is_test, is_backtest, time_frame):
	print('Loading AI...')
	from tensorflow import keras
	#data/models/different_stocks.h5 
	model = keras.models.load_model('data/v2training/0/model', compile=False)
	
	# Load S&P500
	print('Loading stock list...')
	table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	df = table[0]
	sp = df['Symbol']

	sp = sp.tolist()
	sp.remove('AFL') # AFL has wierd data
	sp.remove('DOV') # DOV also did not match historical values
	
	if is_test:
		sp = sp[0:10]


	if is_backtest:
		sp = tradeapi.api.get_data(sp, time_frame)
		
	# setup stocks
	from python.Level1.stock import Stock
	Stock.setup(NUMBARS, model, User.get_api(), time_frame)
	for symbol in sp:
		this_stock = Stock(symbol)
		
	
	
	return Stock, model
	
#===================================================================#
#								Command Line						#
#===================================================================#

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Control Trading AI')
	parser.add_argument("command", metavar="<command>",
						help="'train', 'trade', 'sell', 'test', 'trail', 'log', 'read'")
	
	parser.add_argument("-t", action='store_true', required=False,
						help='Include -t if this is a shortened test')
						
	parser.add_argument("--name", help = "Name for new model when training")
						
	parser.add_argument("-b", action='store_true', required=False)
	
	parser.add_argument("--time", default='1D',
						help = "Time period to buy and sell on")

	parser.add_argument("-p", action='store_true', required=False,
						help='When trading include -f to only trade paper account')
			
	args = parser.parse_args()


	
	#					Run based on arguments
	#------------------------------------------------------------#
	# Backtest
	if args.b:
		if args.command != 'train':
			num_days = input("Enter the number of days to backtest: ")
		
		import python.Level1.Level2.backtest_api as tradeapi
		from python.user import backtestUser as User
		User.update_users(args.p, tradeapi)
		
		Stock, model = import_data(args.t, args.b, args.time)

		if args.command == 'train':
			train(args.name, Stock)
		else:
			backtest(int(num_days), model, Stock)
	else:
	# Everything else
		import alpaca_trade_api as tradeapi
		from python.user import User
		User.update_users(args.p, tradeapi)
		
		if args.command == 'test':
			test()

		elif args.command == 'trade':

			Stock, model = import_data(args.t, args.b, args.time)

			trade(model, Stock)
			
		elif args.command == 'sell':
			quick_sell()
			
		elif args.command == 'trail':
			trailing(args.p)
		
		elif args.command == 'log':
			log()
			
		elif args.command == 'read':
			read()
		else:
			raise InputError("Command must be either 'train', 'run', or 'view'")

