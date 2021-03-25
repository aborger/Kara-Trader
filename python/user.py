from __future__ import print_function

import pickle

import os.path

from googleapiclient.discovery import build

from google_auth_oauthlib.flow import InstalledAppFlow

from google.auth.transport.requests import Request

from python.Level1.stock import Stock

import pandas as pd
import numpy as np

import os
import stripe
import time

PORTFOLIO_HISTORY_DIR = 'data/portfolio_history/'
USER_DATA_DIR = 'data/user_data/'
SENSITIVE_DATA_DIR = '../Sensitive_Data/'

# If modifying these scopes, delete the file token.pickle.

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']



# The ID and range of a sample spreadsheet.

KEYS_SHEET = '1b6aOFfJpCvjXwFaJmqRxIxB_PxVO_eHKWQ6bA5IQB8Q'

KEYS_PAGE = 'Form Responses 1'


class User:
	_users = []
	_boosters = []

	#-----------------------------------------------------------------------#
	#								Initializing							#
	#-----------------------------------------------------------------------#

	def __init__(self, user_info, tradeapi):
		self.info = user_info
		self.sub_id = None
		self.status = True
		print('Loading api...')
		
		#try as live account
		self.api = tradeapi.REST(key_id = self.info['keyID'],
							secret_key = self.info['secret_key'],
							base_url = 'https://api.alpaca.markets')
		try:
			self.api.get_account()
		except tradeapi.rest.APIError:
			self.api = tradeapi.REST(key_id = self.info['keyID'],
						secret_key = self.info['secret_key'],
						base_url = 'https://paper-api.alpaca.markets')
			try:
				self.api.get_account()
			except:
				print(self.info['email'] + ' does not work!')
				self.status = False
			else:
				print(self.info['email'] + ' is paper account')



	@classmethod
	def update_users(cls, is_paper, tradeapi):
		print('Getting users...')
		# reset user list
		cls._users = []
		# gets user data
		user_data = cls._find_users()
		# connects emails to their subscription item
		sub_items = cls._update_stripe_accounts()
		# adds each user to user list
		for user in user_data:
			new_user = User(user, tradeapi)
			email = new_user.info['email']
			if email == 'boost':
				cls._boosters.append(new_user)
			elif is_paper:
				if email == 'kara':
					cls._users.append(new_user)
			else:
				if new_user.status:
					new_user.sub_id = sub_items.get(email, None)
					cls._users.append(new_user)

		
				
	@classmethod
	def _update_stripe_accounts(cls):
		stripe_data = pd.read_csv(SENSITIVE_DATA_DIR + 'Stripe_API.csv', sep=r'\s*,\s*', engine='python')
		stripe_data = stripe_data.iloc[0]

		stripe.api_key = stripe_data["Live Key"]
		
		# Transfer any website subscriptions to new subscription plan
		old_subscriptions = stripe.Subscription.list(price=stripe_data["Old Plan"])
		old_subs = old_subscriptions["data"]
		for sub in old_subs:
			customer_id = sub["customer"]
			subscription_id = sub["id"]
			stripe.Subscription.delete(subscription_id)
			stripe.Subscription.create(customer=customer_id, items=[{"price": stripe_data["New Price"]},],)
			print('New Customer ' + customer_id + 'has been transfered')
		
		# Create list to connect subscription items to emails
		new_subscriptions = stripe.Subscription.list(price=stripe_data["New Price"])
		new_subs = new_subscriptions["data"]
		customer_sub_ids = {}
		for sub in new_subs:
			customer_id = sub["customer"]
			customer = stripe.Customer.retrieve(customer_id)
			email = customer["email"]
			sub_item_id = sub["items"]["data"][0]["id"]
			
			customer_sub_ids[email] = sub_item_id

		return customer_sub_ids

	@classmethod
	def charge_users(cls):
		for user in cls._users:
			if user.sub_id is None:
				print(user.info["email"] + " does not have a subscription!")
			else:
				stripe.SubscriptionItem.create_usage_record(
					user.sub_id,
					quantity = int(float(user.api.get_account().equity) * 100),
					timestamp=int(time.time()))

	@classmethod
	def _find_users(cls):
		creds = None
		# The file token.pickle stores the user's access and refresh tokens, and is
		# created automatically when the authorization flow completes for the first
		# time.


		if os.path.exists(SENSITIVE_DATA_DIR + 'token.pickle'):
			with open(SENSITIVE_DATA_DIR + 'token.pickle', 'rb') as token:
				creds = pickle.load(token)

		# If there are no (valid) credentials available, let the user log in.
		if not creds or not creds.valid:
			if creds and creds.expired and creds.refresh_token:
				creds.refresh(Request())
			else:
				flow = InstalledAppFlow.from_client_secrets_file(SENSITIVE_DATA_DIR + 'credentials.json', SCOPES)
				creds = flow.run_local_server(port=0)

			# Save the credentials for the next run
			with open(SENSITIVE_DATA_DIR + 'token.pickle', 'wb') as token:
				pickle.dump(creds, token)

		service = build('sheets', 'v4', credentials=creds)

		# Call the Sheets API
		sheet = service.spreadsheets()
		result = sheet.values().get(spreadsheetId=KEYS_SHEET, range=KEYS_PAGE).execute()

		user_values = result.get('values', [])

		if not user_values:
			print('No data found.')
		else:
			users = []
			user_values.pop(0)
			for user in user_values:
				user_dict = dict(email=user[1], keyID=user[2], secret_key=user[3])
				users.append(user_dict)
			return users

	
	#-----------------------------------------------------------------------#
	#								Individual								#
	#-----------------------------------------------------------------------#

	def get_equity(self):
		account = self.api.get_account()
		equity = account.equity
		return equity
			
	def get_status(self):
		account = self.api.get_account()
		status = account.status
		return status

	def get_buying_power(self):
		account = self.api.get_account()
		buying_power = account.buying_power
		return buying_power

	def get_gain(self):
		account = self.api.get_account()
		try:
			gain = float(account.equity) / float(account.last_equity)
			gain = (gain - 1) * 100
		except ZeroDivisionError:
			#print(self.info['email'] + ' account is empty')
			pass
		except:
			raise
		else:
			formatted_gain = "{:.2f}".format(gain)
			return formatted_gain

	#-----------------------------------------------------------------------#
	#									Trading								#
	#-----------------------------------------------------------------------#

	@classmethod
	def users_buy(cls, diversified_stocks, NUM_BEST_STOCKS):
		DEBUG = True
		# Buy stocks for each user
		for user in cls._users:
			print('User: ' + user.info['email'])
			print('                 Buying   ')
			print('========================================')
			# find cheapest stock to make sure we buy as much as possible

			buying_power = float(user.api.get_account().buying_power)
			# Buy the recommended ratio of stocks
			for stock_dict in diversified_stocks:
				dollars_to_spend = buying_power * stock_dict['buy_ratio']
				dollars_to_spend = str(round(dollars_to_spend, 2))
				if DEBUG:
					print('------------------------------------------')
					print('Stock: ' + stock_dict['stock_object'].symbol)
					print('gain: ' + str(stock_dict['stock_object'].gain))
					print('buying_power: ' + str(buying_power))
					print('buy_ratio: ' + str(stock_dict['buy_ratio']))
					print('dollars_to_spend: ' + dollars_to_spend)


				stock_dict['stock_object'].buy_notional(user.api, dollars_to_spend)


				

	@classmethod
	def users_sell_all(cls):
		print('                Selling')
		print('========================================')
		for user in cls._users:
			portfolio = user.api.list_positions()
			for position in portfolio:
				stock_position = Stock(position)
				stock_position.sell(user.api, position.qty)

	@classmethod
	def users_trailing(cls):
		print('Setting trailing stop for all users')
		for user in cls._users:
			percent = 1
			if user.info['email'] == 'davidpaper@gmail.com':
				percent = 0.75
			if user.info['email'] == 'davidgoretoy123@gmail.com':
				percent = 0.5
			portfolio = user.api.list_positions()
			for position in portfolio:
				stock_position = Stock(position)
				stock_position.trailing_stop(user.api, position.qty, percent)
				
	@classmethod
	def cancel_orders(cls):
		for user in cls._users:
			pass
			#user.api.cancel_all_orders()


	#-----------------------------------------------------------------------#
	#									Logging								#
	#-----------------------------------------------------------------------#
	@classmethod
	def log(cls, LOGDIR):

		print('logging...')
		for user in cls._users:
			if not os.path.exists(LOGDIR + user.info['email'] + '.csv'):
				# Start log
				log = open(LOGDIR + user.info['email'] + '.csv','w')
				log.write('Time, Equity\n')
				log.close()
			
			# Add to log
			log = open(LOGDIR + user.info['email'] + '.csv', 'a')
			
			# Time
			current_time = cls.get_time()
			log.write(current_time + ', ')
			
			# Equity
			log.write(str(user.get_equity()) + '\n')
			
			log.close()
		
		# Get average gain

		gain_sum = 0.0
		for user in cls._users:
			print(user.info['email'])
			log = pd.read_csv(LOGDIR + user.info['email'] + '.csv', sep=r'\s*,\s*', engine='python')
			log = log.to_numpy()
			end = len(log) - 1
			previous = end - 1
			try:
				gain = float(log[end][1]) / float(log[previous][1])
				gain = (gain -1)
			except ZeroDivisionError:
				print(user.info['email'] + ' is empty')
				gain = 0
				pass
			except:
				raise
			if gain != 'nan':
				gain_sum += gain
			
		avg_gain = gain_sum / len(cls._users)
		
		# Adding average gain to 1main
		# Getting previous account value
		if not os.path.exists(LOGDIR + '1main' + '.csv'):
			# Start log
			log = open(LOGDIR + '1main' + '.csv','w')
			log.write('Time, Average Gain, 1G Account\n,,1000\n')
			log.close()
		logr = pd.read_csv(LOGDIR + '1main.csv', sep=r'\s*,\s*', engine='python')
		logr = logr.to_numpy()
		end = len(logr) -1
		last_equity = logr[end][2]
		
		logw = open(LOGDIR + '1main.csv', 'a')
		# Time
		t = time.localtime()
		current_time = time.strftime("%m/%d/%Y %H:%M:%S", t)
		logw.write(current_time + ', ')
		
		# Gain
		logw.write(str(avg_gain) + ', ')
		
		# Average Account
		new_equity = last_equity + last_equity * avg_gain
		logw.write(str(new_equity) + '\n')				

	@classmethod
	def graph(cls, file):
		import matplotlib
		log = pd.read_csv(file, sep=r'\s*,\s*', engine='python')
		log = log.to_numpy()
		print('Graphing...')
		
		matplotlib.use('TkAgg')
		import matplotlib.pyplot as plt
		plt.plot(log[:,1], color = 'green', label = 'Equity')
		plt.title('Account')
		plt.xlabel('Time')
		plt.ylabel('Account Value')
		plt.legend()
		#plt.savefig('data/results/stock' + str(stock) + '/training' + str(ID) + '.png')
		plt.show()

	@classmethod
	def view(cls, LOGDIR):
		#for user in User._users:
		#LOGDIR + self.info['email'] + '.csv'
		graph(LOGDIR + '1main.csv')

	@classmethod
	def log_portfolio_history(cls):
		num_days = None

		for user in cls._users:
			history = user.api.get_portfolio_history(period = '1A')
			num_days = len(history.timestamp)

			# Prepare to record data
			log = open(PORTFOLIO_HISTORY_DIR + user.info["email"] + '.csv', 'w')	
			log.write('Timestamp, Equity\n')
			
			# Record data
			for day in range(0, num_days):
				daytime = time.strftime('%Y-%m-%d', time.localtime(history.timestamp[day]))
				log.write(str(daytime) + ', ' + str(history.equity[day]) + '\n')

			log.close()	

		Stock.log_bars('SPY', num_days)

	#-----------------------------------------------------------------------#
	#									Getters								#
	#-----------------------------------------------------------------------#

	@classmethod
	def get_api(cls):
		return cls._users[0].api
		
	@classmethod
	def get_api_list(cls):
		apis = []
		for user in cls._users:
			apis.append(user.api)
		return apis

	@classmethod
	def get_boosters(cls):
		apis = []
		for user in cls._users:
			apis.append(user.api)
		for booster in cls._boosters:
			apis.append(booster.api)
		return apis
		
	@classmethod
	def get_Users(cls):
		return cls._users
		
	@classmethod
	def get_stats(cls):
		print('=========================================================')
		for user in cls._users:
			# Status
			print(user.info['email'] + ' is ' + str(user.get_status()))
		print('=========================================================')
		for user in cls._users:
			# Equity
			print(user.info['email'] + ' has equity of ' + str(user.get_equity()))
		print('=========================================================')
		for user in cls._users:
			# Gain
			print(user.info['email'] + ' has gained ' + str(user.get_gain()) + '%')
			
	@classmethod
	def get_time(cls):
		t = time.localtime()
		current_time = time.strftime("%d/%m/%Y %H:%M:%S", t)
		return current_time

	
			
	
#-----------------------------------------------------------------------#
#									BackTest							#
#-----------------------------------------------------------------------#

class backtestUser(User):
	@classmethod
	def get_users(cls):
		users = []
		user_dict = dict(email='BackTestUser1', keyID='BackTest1', secret_key=1000)
		users.append(user_dict)
		user_dict = dict(email='BackTestUser2', keyID='BackTest2', secret_key=10000)
		#users.append(user_dict)
		return users
	'''
	
	@classmethod
	# went to 3/16/2018
	def set_time(cls, day=2, month=1, year=2019, hour=1, minute=1, second=1):
		for user in cls._users:
			user.api.get_clock().set_time(day=day, month=month, year=year, hour=hour, minute=minute, second=second)
	'''
	@classmethod
	def next_day(cls):
		for user in cls._users:
			user.api.next_day()

	@classmethod
	def get_portfolio(cls):
		for user in cls._users:
			print('       ' + user.info['email'])
			print('-------------------------------')
			print('Last: ' + str(user.api.get_account().last_equity))
			print('Equity: ' + str(user.api.get_account().equity))
			positions = user.api.list_positions()
			for position in positions:
	
				print('--------------')
				print(position.symbol)
				print('QTY: ' + str(position.qty))
				print('Price: ' + str(position.current_price))
				print('Value: ' + str(position.qty * position.current_price))

	@classmethod
	def get_time(cls):
		return str(cls._users[0].api.get_clock().days_past - 10)
		
	@classmethod
	def get_stats(cls):
		print('=========================================================')
		for user in cls._users:
			# Status
			print(user.info['email'] + ' is ' + str(user.get_status()))
		print('=========================================================')
		for user in cls._users:
			# Equity
			print(user.info['email'] + ' has equity of ' + str(user.get_equity()))
			'''
		print('=========================================================')
		for user in cls._users:
			# Gain
			print(user.info['email'] + ' has buying power of ' + str(user.get_buying_power()))
			'''
		print('=========================================================')
		for user in cls._users:
			# Gain
			print(user.info['email'] + ' has gained ' + str(user.get_gain()) + '%')
		
		
#-----------------------------------------------------------------------#
#							Demonstration User							#
#-----------------------------------------------------------------------#
class DemonstrationUser(User):
	def __init__(self, user_info, tradeapi):
		self.info = user_info
		self.sub_id = None
		self.status = True
		print('Loading api...')
		
		#try as live account
		self.api = tradeapi.REST(key_id = self.info['keyID'],
							secret_key = self.info['secret_key'],
							base_url = 'https://api.alpaca.markets')
		try:
			self.api.get_account()
		except tradeapi.rest.APIError:
			self.api = tradeapi.REST(key_id = self.info['keyID'],
						secret_key = self.info['secret_key'],
						base_url = 'https://paper-api.alpaca.markets')
			try:
				self.api.get_account()
			except:
				print(self.info['email'] + ' does not work!')
				self.status = False
			else:
				print(self.info['email'] + ' is paper account')

	@classmethod
	def update_users(cls, tradeapi):
		print('Getting demonstration users...')
		# reset user list
		cls._users = []
		# gets user data
		users = []
		if os.path.exists(USER_DATA_DIR + 'demo_user.csv'):
			data = pd.read_csv(DEMO_USER_PATH + 'demo_user.csv', sep=r'\s*,\s*', engine='python')
			data = data.to_numpy()
			for user in data:
				user_dict = dict(email=user[0], keyID=user[1], secret_key=user[2])
				users.append(user_dict)
		else:
			import webbrowser
			print("Please open a free alpaca paper trading account to use demonstration.")
			webbrowser.open('https://app.alpaca.markets/signup')
			keyID = input("Please enter your alpaca API Key ID: ")
			secret_key = input("Please enter your alpaca Secret Key: ")

			# Saving info to file
			demofile = open(DEMO_USER_PATH, 'w')	
			demofile.write('email, keyID, secret_key\n')
			demofile.write('Demo user, ' + keyID + ', ' + secret_key + '\n')
			demofile.close()

			user_dict = dict(email='Demo user', keyID=keyID, secret_key=secret_key)
			users.append(user_dict)


		# adds each user to user list
		for user in users:
			new_user = User(user, tradeapi)
			cls._users.append(new_user)