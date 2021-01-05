import train_rnn

main_nn = DQN()
target_nn = DQN()

optimizer = keras.optimizers.Adam(1e-4)
mse = keras.losses.MeanSquaredError()