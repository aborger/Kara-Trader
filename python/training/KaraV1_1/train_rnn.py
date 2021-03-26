import tensorflow as tf


class config:
    NUMBARS = 10
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    mse = tf.keras.losses.MeanSquaredError()


# ------------------ Model --------------
class RNN(tf.keras.Model):
	def __init__(self):
		super(RNN, self).__init__()
		tf.keras.backend.set_floatx('float64')

		self.LSTM1 = tf.keras.layers.LSTM(units=50,return_sequences=True, input_shape=(config.NUMBARS, 5))
		self.LSTM2 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM3 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM4 = tf.keras.layers.LSTM(units=50,)
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.dense1 = tf.keras.layers.Dense(50)
		self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')


	def call(self, input):
		x = self.LSTM1(input)
		x = self.dropout(x)
		x = self.LSTM2(x)
		x = self.dropout(x)
		x = self.LSTM3(x)
		x = self.dropout(x)
		x = self.LSTM4(x)
		x = self.dropout(x)
		x = self.dense1(x)
		x = self.dense2(x)
		print('output: ' + str(x))
		return x

	def train(self, prediction, truth):
		print('pred shape: ' + str(prediction.shape))
		print('truth shape: ' + str(truth.shape))
		with tf.GradientTape() as tape:
			pred = self.call(prediction)
			print('pred: ' + str(pred))
			loss = config.mse(truth, pred)
		grads = tape.gradient(loss, self.trainable_variables)
		config.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return loss
