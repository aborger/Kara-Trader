# This file manages the other files

# Number of bars to predict on
# Ex: If NUMBARS=4 use monday-thursday to predict friday 

import pandas as pd
import time

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
def train(name):
	import python.training.head_class as hc
	hc.Training_Model.oversee(TRAINSET, TESTSET, MODELS, name)
	
def backtest(numdays, model, Stock):
	for day in range(0, numdays):
		log()
		trade(model, Stock)
		for user in User.get_User():
			user.api.get_account().remove_empty()
			
		User.get_portfolio()
		User.next_day()
		print('                       Next Day')
		print('=======================================================')
		User.get_portfolio()

def test():
	User.get_stats()
	
def charge():
    import stripe
    from time import time
    TEST_KEY = "sk_test_51HLFZMEZVv1JoaylrbkPZTBcUTiq9QMbxeyRTYd4rncGS5NZFCEdhtEJftz8LpM7Mj7g8NXKEMQXEurCd2R0RR5y00r4KGsXAM"
    LIVE_KEY = "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"
    stripe.api_key = LIVE_KEY

    for user in User.get_User():
        '''
        if user.info["email"] == 'test':
            stripe.SubscriptionItem.create_usage_record(
				"si_IQaqXjwSodgl02",
				quantity = int(float(user.api.get_account().equity) * 100),
				timestamp=int(time()))
    
        elif user.info["email"] == 'aborger@nnu.edu':
            stripe.SubscriptionItem.create_usage_record(
                "si_IQbjimY9WtxZbl",
				quantity = int(float(user.api.get_account().equity) * 100),
                timestamp=int(time()))
        '''
        if user.info["email"] == 'davidgoretoy123@gmail.com':
            stripe.SubscriptionItem.create_usage_record(
                "si_IfslYjzpuV2Bde",
				quantity = int(float(user.api.get_account().equity) * 100),
				timestamp=int(time()))

def log():
	User.log(LOGDIR)
	
def read():
	User.view(LOGDIR)
	
def quick_sell():

	# Sell any open positions
	User.users_sell()
	
def trailing(is_paper):
    User.users_trailing()

def trade(Stock, User, model):
	NUM_BEST_STOCKS = 5
	#from python.stock import Stock
	#from python.PYkeys import Keys
	from time import sleep


	if True: #User.get_api().get_clock().is_open:
		User.cancel_orders()
		# Sell any open positions
		User.users_sell_all()
		# Gets best stocks and ratio of how many to buy
		print('Calculating best stocks...')

		start_time = time.time()
		diversified_stocks, best_stocks = Stock.find_diversity(NUM_BEST_STOCKS)
		end_time = time.time()

		print(f"Time to predict is {end_time - start_time}")

		# Buy the best stocks
		User.users_buy(diversified_stocks, best_stocks)
	else:
		print('Stock market is not open today.')
		
#===================================================================#
#							Helping Functions						#
#===================================================================#
def import_data(is_test, is_backtest, time_frame):
	print('Loading AI...')
	from tensorflow import keras
	model = keras.models.load_model('data/models/different_stocks.h5', compile=False)
	
	if is_backtest:
		from python.user import backtestUser as User
		User.update_users(is_test, tradeapi)
	else:
		from python.user import User
		User.update_users(is_test, tradeapi)

	# setup stocks
	from python.Level1.stock import Stock
	Stock.setup(NUMBARS, model, time_frame, User.get_api())


	# Load S&P500
	print('Loading stock list...')
	
	table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	df = table[0]
	sp = df['Symbol']

	sp = sp.tolist()
	sp.remove('AFL') # AFL has wierd data
	sp.remove('DOV') # DOV also did not match historical values
	
	if is_test:
		sp = sp[0:20]


	if is_backtest:
		sp = tradeapi.api.get_data(sp, time_frame)
	
	for symbol in sp:
		this_stock = Stock(symbol)
	

	'''
	import alpaca_trade_api as theapi
	# get all available stocks
	assets = User.get_api().list_assets(status='active')
	for asset in assets:
		this_stock = Stock(asset.symbol)
	'''
	
	
	
	return Stock, User, model
	
#===================================================================#
#								Command Line						#
#===================================================================#

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Control Trading AI')
	parser.add_argument("command", metavar="<command>",
						help="'train', 'trade', 'sell', 'test', 'trail', 'log', 'read', 'charge'")
	
	parser.add_argument("-t", action='store_true', required=False,
						help='Include -t if this is a shortened test')
						
	parser.add_argument("--name", help = "Name for new model when training")
						
	parser.add_argument("-b", action='store_true', required=False)
	
	parser.add_argument("--time", default='1D',
						help = "Time period to buy and sell on")

	parser.add_argument("-p", action='store_true', required=False,
						help='When trading include -p to only trade paper account')
			
	args = parser.parse_args()


	
	#					Run based on arguments
	#------------------------------------------------------------#
	# Backtest
	if args.b:
		num_days = input("Enter the number of days to backtest: ") 
		import python.Level1.Level2.backtest_api as tradeapi
		
		
		Stock, User, model = import_data(args.t, args.b, args.time)

		backtest(int(num_days), model, Stock)
	else:
	# Everything else
		import alpaca_trade_api as tradeapi
		Stock, User, model = import_data(args.t, args.b, args.time)
		
		if args.command == 'train':
			train(args.name)

		elif args.command == 'test':
			test()

		elif args.command == 'buy':
			trade(Stock, User, model)
		
		elif args.command == 'charge':
			charge()
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

