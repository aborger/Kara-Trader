from python.time_frame import Time_frame

class Stock:
    _stocks = []
    owned = []
    _api = 0
    
    def setup(NUMBARS, model, api):
        Time_frame.setup(NUMBARS, model, api)
        Stock._api = api
        
    
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
    
    def highest_gain(time_frame): # returns 5 best stocks
        time_frame = Stock._convert_frame_name(time_frame)
                
        def get_gain(stock):
                return stock.frames[time_frame].gain
        
        # Currently only using 5 max gains
        print("Getting highest gain...")
        max_stocks = []
        for stock in Stock._stocks:
            stock.frames[time_frame].get_gain()
            print(stock.symbol + "'s gain is " + str(stock.frames[time_frame].gain))
            if len(max_stocks) < 5:
                max_stocks.append(stock)
            elif stock.frames[time_frame].gain > max_stocks[4].frames[time_frame].gain:
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
    def return_gain(self, time_frame):
        time_frame = Stock._convert_frame_name(time_frame)
        return self.frames[time_frame].gain
    
    # updates gain and returns it
    def get_gain(self, time_frame):
        time_frame = Stock._convert_frame_name(time_frame)
        self.frames[time_frame].get_gain()
        return self.frames[time_frame].gain
    
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
        


        



    
