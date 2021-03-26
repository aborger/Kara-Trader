from .train_rnn import RNN
from python.Level1.Level2.predict import norm, denorm, fit
import numpy as np

def train(Stock, NUMBARS, TRAIN_BAR_LENGTH, DATA_PER_STOCK):
    print('collecting prices...')
    Stock.collect_prices(TRAIN_BAR_LENGTH)


    print('converting to dataset...')
    dataset = [stock.prev_bars for stock in Stock.get_stock_list()]
    npSet = np.array(dataset, dtype=object)

    batches = np.empty(shape=(npSet.shape[0], TRAIN_BAR_LENGTH, NUMBARS, DATA_PER_STOCK))
    truths = np.empty(shape=(npSet.shape[0], TRAIN_BAR_LENGTH))

    for stock_num in range(0, npSet.shape[0]):
        for i in range(NUMBARS, TRAIN_BAR_LENGTH):
            batch = npSet[stock_num,i-NUMBARS:i]
            batches[stock_num, i] = batch
            truth = npSet[stock_num,i,2]
            truths[stock_num, i] = truth

    print('batches: ' + str(batches.shape))
    print('truths: ' + str(truths.shape)) 
    model = RNN()

    print('training...')
    for batch_num in range(0, batches.shape[0]):
        normal, key = norm(batches[batch_num])
        norm_truth = fit(truth[batch_num], key)

        loss = model.train(normal, norm_truth)
        print('Batch ' + str(batch_num) + '. Loss of ' + str(loss))

    

