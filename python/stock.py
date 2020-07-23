import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Stock:
    
    def __init__(self, symbol, NUMBARS, api):
        self.symbol = symbol
        self.NUMBARS = NUMBARS
        self.api = api
        
    def predictions(self, model):
        self.prediction_list = [self.get_prediction('1Min', model),
                            self.get_prediction('5Min', model),
                            self.get_prediction('15Min', model),
                            self.get_prediction('1D', model)]
        
    def get_current_price(self):
        barset = (self.api.get_barset(self.symbol,'1Min',limit=1))
        symbol_bars = barset[self.symbol]
        current_price = symbol_bars[0].c
        return current_price
        
    def get_max_gain(self, model):
        self.predictions(model)
        
        current_price = self.get_current_price()
        
        # predict percent gain
        highest_prediction = self.prediction_list.index(max(self.prediction_list))
        gain = self.prediction_list[highest_prediction]/current_price
        gain = round((gain -1) * 100, 3)
        
        self.max_gain = gain
        print('Gain for ' + self.symbol + ' is ' + str(gain) + '%')
        
    def get_1D_gain(self, model):
        prediction = self.get_prediction('1D', model)
        current = self.get_current_price()
        
        gain = prediction/current
        gain = round((gain -1) * 100, 3)
        print('Gain for ' + self.symbol + ' is ' + str(gain))

    # time_frame can be 1Min, 5Min, 15Min, or 1D
    def get_prediction(self, time_frame, model):
        #print('Getting prediction for ' self.symbol ' on ' + time_frame + ' time frame')
        # Get bars
        barset = (self.api.get_barset(self.symbol,time_frame,limit=self.NUMBARS))
        # Get symbol's bars
        symbol_bars = barset[self.symbol]

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
        reshapedSet = np.reshape(npDataSet, (1, self.NUMBARS, 5))
        
        # Normalize Data
        sc = MinMaxScaler(feature_range=(0,1))
        normalized = np.empty(shape=(1, self.NUMBARS, 5)) 
        normalized[0] = sc.fit_transform(reshapedSet[0])
        
        # Predict Price
        predicted_price = model.predict(normalized)
        
        
        # Add 4 columns of 0 onto predictions so it can be fed back through sc
        shaped_predictions = np.empty(shape = (1, 5))
        for row in range(0, 1):
            shaped_predictions[row, 0] = predicted_price[row, 0]
        for col in range (1, 5):
            shaped_predictions[row, col] = 0
        
        
        # undo normalization
        predicted_price = sc.inverse_transform(shaped_predictions)

        return predicted_price[0][0]
