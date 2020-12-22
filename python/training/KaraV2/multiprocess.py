import multiprocessing
import random
from time import sleep

NUM_STOCKS = 10000

#print(multiprocessing.cpu_count())


def get_best(stocks):
    while True:
        print('--------------------------')
        for stock in range(0, len(stocks)):
            rand = random.randint(0, 99)
            stocks[stock] = rand

def choose_best(stocks):
    while True:
        best_stock = 0
        best = -1
        for stock_num in range(0, len(stocks)):
            
            if stocks[stock_num] > best:
                
                best = stocks[stock_num]
                best_stock = stock_num
        print('Stock Number: ' + str(best_stock) + ' Score: ' + str(stocks[best_stock]))

if __name__ == '__main__':
    stocks = multiprocessing.Array('i', NUM_STOCKS)

    p1 = multiprocessing.Process(target=get_best, args=(stocks,))
    p2 = multiprocessing.Process(target=choose_best, args=(stocks,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
