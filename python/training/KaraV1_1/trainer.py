from .train_rnn import RNN
from python.Level1.Level2.predict import norm, denorm, fit, pnp, tpnp
import numpy as np


def train(Stock, NUMBARS, TRAIN_BAR_LENGTH, DATA_PER_STOCK):

    print('collecting prices...')
    Stock.collect_prices(TRAIN_BAR_LENGTH) # EMXC and FTCH cause issues

    print('converting to dataset...')
    dataset = [stock.prev_bars for stock in Stock.get_stock_list()]
    npSet = np.array(dataset, dtype=object)

    #tpnp('EMXC', npSet[335])
    
    #pnp('FTCH', npSet[467])

    clusters = [] # each cluster is what the model uses to make one prediction for one stock
    truths = []

    for stock in npSet:
        for i in range(NUMBARS, TRAIN_BAR_LENGTH):
            cluster = stock[i-NUMBARS:i]
            clusters.append(cluster)
            truth = stock[i,2]
            truths.append(truth)

    clusters = np.array(clusters)
    truths = np.array(truths)

    model = RNN()

    num_batches = npSet.shape[0]
    batch_size = int(clusters.shape[0] / num_batches)
    print('Number of batches: ' + str(num_batches)) # each stock's data is a batch
    print('Batch size: ' + str(batch_size))
    for batch_num in range(0, num_batches - 1):
        clus_i = batch_num * batch_size # starting cluster
        clus_f = clus_i + batch_size # ending cluster
        normal, key = norm(clusters[clus_i : clus_f])

        norm_truth = fit(truths[clus_i : clus_f], key)
        normal = normal.astype('float32')
        norm_truth = norm_truth.astype('float32')

        loss = model.train(normal, norm_truth)
        print('Batch ' + str(batch_num) + '. Loss of ' + str(loss))


    print('Validating...')
    clus_i = (num_batches - 1) * batch_size
    clus_f = clus_i + batch_size

    normal, key = norm(clusters[clus_i : clus_f])
    real_price = truths[clus_i : clus_f]
    normal = normal.astype('float32')

    pred = model(normal)
    predicted_price = denorm(pred, key)


    print('Results: ')
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.plot(real_price, color = 'black', label = 'Stock Price')
    plt.plot(predicted_price, color = 'green', label = 'Price Prediction')
    plt.title('Prediction')
    plt.xlabel('Bars')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
