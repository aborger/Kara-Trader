from python.time_frame import Time_frame

class Stock:
    _stocks = []
    owned = []
    _api = 0
    _period = 0
    
    def setup(NUMBARS, model, api, period):
        Time_frame.setup(NUMBARS, model, api)
        Stock._api = api
        Stock._period = Stock._convert_frame_name(period)
        
    
    def _convert_frame_name(time_frame):
        if time_frame == '1Min':
            time_frame = 0
        elif time_frame == '5Min':
            time_frame = 1
        elif time_frame == '15Min':
            time_frame = 2
        elif time_frame == '1D':
            time_frame = 3
        else:
            raise InputError('Incorrect time frame')
        return time_frame
    
    def highest_gain(num_stocks): # returns num_stocks best stocks
                
        def get_gain(stock):
                return stock.frames[Stock._period].gain
        
        # Currently only using 5 max gains
        print("Getting highest gain...")
        max_stocks = []
        for stock in Stock._stocks:
            stock.frames[Stock._period].get_gain()
            print(stock.symbol + "'s gain is " + str(stock.frames[Stock._period].gain))
            if len(max_stocks) < num_stocks:
                max_stocks.append(stock)
            elif stock.frames[Stock._period].gain > max_stocks[num_stocks - 1].frames[Stock._period].gain:
                max_stocks.pop()
                max_stocks.append(stock)
            
            # sort list so lowest gain is at the end
            max_stocks.sort(reverse=True, key=get_gain)
        return max_stocks
        
    def __init__(self, symbol):
        self.symbol = symbol
        self._stocks.append(self)
        
        self.frames = [Time_frame('1Min', symbol), Time_frame('5Min', symbol),
                        Time_frame('15Min', symbol), Time_frame('1D', symbol)]
        
    # returns saved gain
    def return_gain(self):
        return self.frames[Stock._period].gain
    
    # updates gain and returns it
    def get_gain(self):
        self.frames[Stock._period].get_gain()
        return self.frames[Stock._period].gain
    
    def buy(self):
        Stock.owned.append(self)
        print ('Bought ' + self.symbol)
        Stock._api.submit_order(
            symbol=self.symbol,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc')
        
    def sell(self):
        Stock.owned.remove(self)
        print ('Sold ' + self.symbol)
        Stock._api.submit_order(
            symbol=self.symbol,
            qty=1,
            side='sell',
            type='market',
            time_in_force='gtc')
        


        



    
