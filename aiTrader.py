# Dependencies:
# pip3 install alpaca-trade-api
import alpaca_trade_api as tradeapi
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

NUMSTOCKS = 100
NUMBARS = 10

# Login to Alpaca
api = tradeapi.REST(key_id='PKW854NSAYD72P0XLDJJ',
	secret_key='5BMSm7xg9QvkjGssEMdzTg5yjeAYj4S2OHIRLF6q',
	base_url='https://paper-api.alpaca.markets')

# Load Model
model = keras.models.load_model('Trade-Model')

# Load S&P500
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
sp = df['Symbol']

# Predict difference for each stock
def FindDifferences():
  predicted_differences = []
  for symbol in range(0, NUMSTOCKS):
    # Get bars
    barset = (api.get_barset(sp[symbol],'5Min',limit=10))
    # Get symbol's bars
    symbol_bars = barset[sp[symbol]]
    print(sp[symbol])
    # Convert to list
    dataSet = []
    for barNum in symbol_bars:
      dataSet.append(barNum.o)
    # Convert to numpy array
    npDataSet = np.array(dataSet)
    reshapedSet = np.reshape(npDataSet, (1, NUMBARS, 1))
    # Normalize Data
    sc = MinMaxScaler(feature_range=(0,1))
    normalized = np.empty(shape=(1, NUMBARS, 1)) 
    normalized[0] = sc.fit_transform(reshapedSet[0])
    # Predict Price
    predicted_price = model.predict(normalized)
    # undo normalization
    predicted_price = sc.inverse_transform(predicted_price)
    # add difference to array
    difference = predicted_price[0,0] - reshapedSet[0, NUMBARS - 1, 0]
    predicted_differences.append(difference)
  return predicted_differences

# Buy Stock
def BuyStock(stock):
  print ('Bought ' + stock)
  api.submit_order(
    symbol=stock,
    qty=1,
    side='buy',
    type='market',
    time_in_force='gtc')
	
# Sell Stock
def SellStock(stock):
  print ('Sold ' + stock)
  api.submit_order(
    symbol=stock,
    qty=1,
    side='sell',
    type='market',
    time_in_force='gtc')

# Main
predicted_differences = FindDifferences()
best_stock = sp[predicted_differences.index(min(predicted_differences))]
BuyStock(best_stock)
time.sleep(30)
SellStock(best_stock)