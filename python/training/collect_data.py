
import alpaca_trade_api as tradeapi
# get s&p500
import pandas as pd
def collect(api, STOCK_NUM, type, TRAINBARLENGTH, Time):
	table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	df = table[0]
	sp = df['Symbol']

	#print(sp)

	import os, os.path

	barset = []
	for symbol in range(STOCK_NUM, STOCK_NUM + 1):
	  barset.append(api.get_barset(sp[symbol],Time,limit=TRAINBARLENGTH))
	  print(sp[symbol])

	#print(barset[0])

	if type == 'train':
		Dataset = open(r"data/dataset.csv",'w')
	elif type == 'test':
		Dataset = open(r"data/testSet.csv",'w')

	Dataset.write('Open, Close, High, Low, Volume\n')

	for symbolNum in range(STOCK_NUM, STOCK_NUM + 1):
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

