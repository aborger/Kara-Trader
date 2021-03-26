from .train_rnn import RNN
from python.Level1.Level2.predict import norm, denorm, fit
import numpy as np

def train(Stock, NUMBARS, TRAIN_BAR_LENGTH, DATA_PER_STOCK):
    print('collecting prices...')
    Stock.collect_prices(TRAIN_BAR_LENGTH)


    print('converting to dataset...')
    dataset = [stock.prev_bars for stock in Stock.get_stock_list()]
    npSet = np.array(dataset, dtype=object)
    print('npSet: ' + str(npSet.shape))
    print(npSet)

    batches = []
    truths = []

    for stock in npSet:
        for i in range(NUMBARS, TRAIN_BAR_LENGTH):
            batch = stock[i-NUMBARS:i]
            batches.append(batch)
            truth = stock[i,2]
            truths.append(truth)

    batches = np.array(batches)
    truths = np.array(truths)

    print('batches ' + str(batches.shape))
    print(batches)

    print('truths ' + str(truths.shape))
    print(truths)

    model = RNN()

    print('training...')
    for batch_num in range(0, batches.shape[0]):
        normal, key = norm(batch[batch_num])
        norm_truth = fit(truths[batch_num], key)

        loss = model.train(normal, norm_truth)
        print('Batch ' + str(batch_num) + '. Loss of ' + str(loss))

    

