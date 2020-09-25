from __future__ import print_function

import pickle

import os.path

from googleapiclient.discovery import build

from google_auth_oauthlib.flow import InstalledAppFlow

from google.auth.transport.requests import Request

import alpaca_trade_api as tradeapi

from ..stock import Stock

# If modifying these scopes, delete the file token.pickle.

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']



# The ID and range of a sample spreadsheet.

#SAMPLE_SPREADSHEET_ID = '1LJogRzysqMesYmIL__1Rg2FNP4BfNivQMaZLw_bvu6M'
SAMPLE_SPREADSHEET_ID = '1b6aOFfJpCvjXwFaJmqRxIxB_PxVO_eHKWQ6bA5IQB8Q'

SAMPLE_RANGE_NAME = 'Form Responses 1'


class User:
	_users = []
	def update_users():
		print('Getting users...')
		# reset user list
		User._users = []
		# gets user data
		user_data = get_users()
		# adds each user to user list
		for user in user_data:
			new_user = User(user)
			User._users.append(new_user)
	def get_api():
		return User._users[0].api
		
	def get_equities():
		for user in User._users:
			account = user.api.get_account()
			equity = account.equity
			print(user.info['email'] + ' has equity of ' + str(equity))
			
	def get_status():
		for user in User._users:
			account = user.api.get_account()
			status = account.status
			print(user.info['email'] + ' is ' + status)
	
	def get_gain():
		for user in User._users:
			account = user.api.get_account()
			try:
				gain = float(account.equity) / float(account.last_equity)
			except ZeroDivisionError:
				print(user.info['email'] + ' account is empty')
				pass
			except:
				raise
			else:
				print(user.info['email'] + ' has gained ' + str(gain) + '%')
			
	def users_buy(best_stocks):
		# Buy stocks for each user
		for user in User._users:
			print('User: ' + user.info['email'])
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
					debug = True
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
				
	def users_sell():
		print('All users are selling...')
		for user in User._users:
			portfolio = user.api.list_positions()
			for position in portfolio:
					Stock.sell_named_stock(position.symbol, user.api, position.qty)
		
	def __init__(self, user_info):
		self.info = user_info
		print('Loading api...')
		# if aborger then paper-api
		if self.info['email'] == 'test':
			self.api = tradeapi.REST(key_id = self.info['keyID'],
						secret_key = self.info['secret_key'],
						base_url = 'https://paper-api.alpaca.markets')
		else:
			self.api = tradeapi.REST(key_id = self.info['keyID'],
							secret_key = self.info['secret_key'],
							base_url = 'https://api.alpaca.markets')
	

def get_users():

	"""Shows basic usage of the Sheets API.

	Prints values from a sample spreadsheet.

	"""

	creds = None

	# The file token.pickle stores the user's access and refresh tokens, and is

	# created automatically when the authorization flow completes for the first

	# time.
	path_to_user_data = 'python/user_data/'
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
