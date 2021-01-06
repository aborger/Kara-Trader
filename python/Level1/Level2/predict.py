from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras

	
def find_gain(stock, api, model, time_frame, NUMBARS):
    print('Predicting gain for ' + stock.symbol)
    # Get bars
    barset = (api.get_barset(stock.symbol, time_frame, limit=NUMBARS))
    # Get symbol's bars
    symbol_bars = barset[stock.symbol]

    # Convert to list
    dataSet = []

    for barNum in symbol_bars:
        bar = []
        bar.append(barNum.o)
        bar.append(barNum.c)
        bar.append(barNum.h)
        bar.append(barNum.l)
        bar.append(barNum.v)
        dataSet.append(bar)
        
        
    # Convert to numpy array
    npDataSet = np.array(dataSet)
    reshapedSet = np.reshape(npDataSet, (1, NUMBARS, 5))
    
    # Normalize Data
    sc = MinMaxScaler(feature_range=(0,1))
    normalized = np.empty(shape=(1, NUMBARS, 5)) 
    normalized[0] = sc.fit_transform(reshapedSet[0])
    
    # Predict Price
    prediction = model.predict(normalized)
    
    
    # Add 4 columns of 0 onto predictions so it can be fed back through sc
    shaped_predictions = np.empty(shape = (1, 5))
    for row in range(0, 1):
        shaped_predictions[row, 0] = prediction[row, 0]
    for col in range (1, 5):
        shaped_predictions[row, col] = 0
    
    
    # undo normalization
    unshaped_predicted_price = sc.inverse_transform(shaped_predictions)
    predicted_price = unshaped_predicted_price[0][0]

    barset = (api.get_barset(stock.symbol,'1Min',limit=1))
    symbol_bars = barset[stock.symbol]
    current_price = symbol_bars[0].c

        
    gain = predicted_price/current_price
    gain = round((gain -1) * 100, 3)
    if gain < 0:
        gain = 0
    stock.set_gain(gain)
    return stock

def find_gains(worker, time_frame, NUMBARS):
    model = keras.models.load_model('data/models/different_stocks.h5', compile=False)
    stocks = worker["stocks"]
    api = worker["api"]

    predicted_stocks = []
    for stock in stocks:
        predicted_stock = find_gain(stock, api, model, time_frame, NUMBARS)
        predicted_stocks.append(predicted_stock)
    return predicted_stocks


		
