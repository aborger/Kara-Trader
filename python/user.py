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
import time

# If modifying these scopes, delete the file token.pickle.

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']



# The ID and range of a sample spreadsheet.

#SAMPLE_SPREADSHEET_ID = '1LJogRzysqMesYmIL__1Rg2FNP4BfNivQMaZLw_bvu6M'
SAMPLE_SPREADSHEET_ID = '1b6aOFfJpCvjXwFaJmqRxIxB_PxVO_eHKWQ6bA5IQB8Q'

SAMPLE_RANGE_NAME = 'Form Responses 1'


class User:
	_users = []

	@classmethod
	def update_users(cls, is_paper, tradeapi):
		print('Getting users...')
		# reset user list
		cls._users = []
		# gets user data
		user_data = cls.get_users()
		# adds each user to user list
		for user in user_data:
			new_user = User(user, tradeapi)
			if not is_paper:
				if new_user.status:
					cls._users.append(new_user)
			else:
				if new_user.info['email'] == 'test':
					cls._users.append(new_user)
					
				
	@classmethod
	def get_api(cls):
		return cls._users[0].api
		
	# Individual stats
	def get_equity(self):
		account = self.api.get_account()
		equity = account.equity
		return equity
			
	def get_status(self):
		account = self.api.get_account()
		status = account.status
		return status
	
	@classmethod
	def get_User(cls):
		return cls._users
		
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
	
	# Overall stats
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
		
	def view(LOGDIR):
		#for user in User._users:
		#LOGDIR + self.info['email'] + '.csv'
		graph(LOGDIR + '1main.csv')
		
	@classmethod
	def users_buy(cls, best_stocks):
		# Buy stocks for each user
		for user in cls._users:
			#print('User: ' + user.info['email'])
			print('                 Buying   ')
			print('========================================')
			# find cheapest stock to make sure we buy as much as possible
			cheapest_price = 999999
			for stock_dict in best_stocks:
				price = stock_dict['stock_object'].get_current_price()
				if price < cheapest_price:
					cheapest_price = price
			account = user.api.get_account()
			buying_power = int(float(account.buying_power))
			# Repeat buying until its not worth buying anymore
			still_buying = True
			while still_buying:
				spent = 0
				still_buying = False # set to false so if no stocks are bought it is done
				for stock_dict in best_stocks:
					max_money_for_stock = buying_power * stock_dict['buy_ratio']
					current = stock_dict['stock_object'].get_current_price()
					quantity = int(max_money_for_stock / current)
					debug = False
					if debug:
						print('------------------------------------------')
						print('Stock: ' + stock_dict['stock_object'].symbol)
						print('gain: ' + str(stock_dict['stock_object'].get_gain()))
						print('buying_power: ' + str(buying_power))
						print('buy_ratio: ' + str(stock_dict['buy_ratio']))
						print('max_money: ' + str(max_money_for_stock))
						print('Current: ' + str(current))
						print('Quantity: ' + str(quantity))
					if quantity > 0:
						if quantity * current < buying_power:
							stock_dict['stock_object'].buy(user.api, quantity)
							spent += quantity * current
							still_buying = True
						else:
							print('Insufficient buying power')
				buying_power = buying_power - spent
				
				
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
				Stock.trailing_stop(position.symbol, user.api, position.qty, percent)
				
	@classmethod
	def users_sell(cls):
		print('                Selling')
		print('========================================')
		for user in cls._users:
			portfolio = user.api.list_positions()
			for position in portfolio:
				Stock.sell_named_stock(position.symbol, user.api, position.qty)
		
			
	
	
	
	def __init__(self, user_info, tradeapi):
		self.info = user_info
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
	def get_users(cls):

		creds = None

		# The file token.pickle stores the user's access and refresh tokens, and is

		# created automatically when the authorization flow completes for the first

		# time.
		path_to_user_data = 'data/user_data/'
		if os.path.exists(path_to_user_data + 'token.pickle'):

			with open(path_to_user_data + 'token.pickle', 'rb') as token:

				creds = pickle.load(token)

		# If there are no (valid) credentials available, let the user log in.

		if not creds or not creds.valid:

			if creds and creds.expired and creds.refresh_token:

				creds.refresh(Request())

			else:

				flow = InstalledAppFlow.from_client_secrets_file(

					path_to_user_data+ 'credentials.json', SCOPES)

				creds = flow.run_local_server(port=0)

			# Save the credentials for the next run

			with open(path_to_user_data + 'token.pickle', 'wb') as token:

				pickle.dump(creds, token)



		service = build('sheets', 'v4', credentials=creds)



		# Call the Sheets API

		sheet = service.spreadsheets()

		result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,

									range=SAMPLE_RANGE_NAME).execute()

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
	def reset(cls):
		#print('------------ Reset ----------')
		for user in cls._users:
			user.api.reset()
			#print('Equity = ' + str(user.api.get_account().equity))
			
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
		
		
