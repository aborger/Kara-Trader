
import alpaca_trade_api as tradeapi

# get s&p500
import pandas as pd
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
sp = df['Symbol']

print(sp)

import os, os.path

from PYkeys import Keys
keys = Keys()
api = tradeapi.REST(key_id = keys.get_key_id(),
                    secret_key = keys.get_secret_key(),
                    base_url = keys.get_base_url())

barset = []
for symbol in range(503, 504):
  barset.append(api.get_barset(sp[symbol],'5Min',limit=1000))
  print(sp[symbol])

#print(barset[0])

Dataset = open(r"newDataset.csv",'w')

Dataset.write('Open, Close, High, Low, Volume\n')

for symbolNum in range(503, 504):
  symbol = sp[symbolNum]
  symbol_bars = barset[0][symbol] # 0 = symbolNum
  for barNum in symbol_bars:
  #test = symbol
    #print(barNum)
    Dataset.write(str(barNum.t) + ',')
    Dataset.write(str(barNum.o) + ',')
    Dataset.write(str(barNum.c) + ',')
    Dataset.write(str(barNum.h) + ',')
    Dataset.write(str(barNum.l) + ',')
    Dataset.write(str(barNum.v) + '\n')

Dataset.close()

