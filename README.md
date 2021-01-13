<p align="center">
  <img src="https://www.karatrader.com/wp-content/uploads/2020/10/cropped-KARA-individuals-06.png" alt="Kara Logo" width="250">
</p>

# Kara-Trader
[![Build Status](https://travis-ci.com/aborger/AItrader.svg?branch=master)](https://travis-ci.com/aborger/AItrader)

Kara is an artificial intelligence that is capable of predicting stock prices. Currently it uses the past 10 bars of data (Open, Close, High, Low, Volume) to predict the next price. These predictions are uploaded to the [Kara Trader Website](https://www.karatrader.com) as part of the Kara Indicator system.

<h3 align="center">Kara Predictions</h3>
<p align="center">
  <img src="https://www.karatrader.com/wp-content/uploads/2021/01/kara_indicator_table.png" alt="Kara Performance" width="600">
</p>




It calculates the percent gain of a list of stocks (Currently uses the S&P 500) and diversifies amongst the top 5. Diversification is accomplished by calculating a ratio of how funds should be split amongst the top stocks. The amount of funds distributed to a stock is proportional to the gain percentage they are predicted to accomplish. The stocks are bought through the [Alpaca](https://alpaca.markets/) broker. A trailing stop is then placed on the position. The ratio only needs to be calculated once because every user's buying power can be multiplied by the ratio to calculate how many shares of each stock to buy. While some trailing stops may trigger early and the predictions may not be 100% correct, they allow for enough profit enough of the time to offer a considerable profit.


## Performance

<h3 align="center">Real Kara Trader Account</h3>
<p align="center">
  <img src="https://www.karatrader.com/wp-content/uploads/2021/01/kara_performance.png" alt="Kara Performance" width="600">
</p>

As of 1/12/2021 this account that has been trading purely with the Kara AI algorithm has just about doubled in 16 weeks. This account has grown 98.9% while in the exact same time period the S&P 500 has only grown 15.8%. 

A backtest is currently being ran to calculate how accurate Kara's predictions are.

## Known Issues
Smaller accounts don't have the same performance as larger accounts. The issue is caused by smaller accounts not being able to purchase a stock with a value larger than the account's value. An account that started with just $1,000 that ran Kara for the same time period as the $10,000 account that experience a 98.9% growth only experience a 40.8% growth.

## How It Works

#### The Model:
The model is actually very simple. 4 RNN (LSTM) layers with a dense layer at the end.

#### The Training:
The model was trained on 1000 daily bars of every stock in the S&P 500. It uses a simple mean squared error loss function to fit the model.

## Kara V2

This model is a great demonstration of the power of AI as this algorithm only buys at the same time every day and does not even take into account the timing of the market. A second version is in development that takes timing into account. The new version implents a Deep Q Network technique which is a form of reinforcement learning. The environment is the past 10 bars of data. The action space is a buy, sell, or hold option. The reward is the value of the account at the end of the training period. The second version can be found in the working_copy and 2Step_Net branches. The working_copy version was the first implementation of the new techinque. It uses the same model architecture as the original version. The 2Step_Net branch is being used to experiment with a new model architecture involving using two neural networks. The first network is an RNN that uses the individual stock's market data to give an evaluation value (between 0 and 1). Every stock in the lists evaluation value is then fed into a second network along with its buying power/price ratio, and current number of shares in the portfolio. Splitting up the network allows for parallel processing to be used to quickly calculate an evaluation for each stock, as each core can be given a certain number of stocks to process. Then the evaluation data can be fed to the second network. Each user will then have their data fed through the second network along with the evaluations to produce an action.


#### Progress:
So far the second version is coming along great! The system has been built and training has begun. The only issues are mostly due to speed. The entire process is being ran on a raspberry pi 4 which has 4 cores and doesn't even have a GPU. Even with those limitations it takes about 2 minutes to produce an action. However, having the option to buy every 2 minutes is significantly better than only being able to buy once a day.


Checkout the [Kara Trader](https://www.karatrader.com) website for more information!


Note: I am currently modeling a business around Kara Trader. This repository is public to allow viewing of technique, methods, and skills used. The program may still function correctly, but key features have been left out so your experience will not be the same as the functional program.
