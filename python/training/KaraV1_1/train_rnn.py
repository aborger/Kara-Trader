import tensorflow as tf
from python.Level1.Level2.predict import pnp
#tf.keras.backend.set_floatx('float64')

class config:
    NUMBARS = 10
    num_epochs = 1

    optimizer = tf.keras.optimizers.Adam(1e-4)
    mse = tf.keras.losses.MeanSquaredError()


# ------------------ Model --------------
class RNN(tf.keras.Model):
	def __init__(self):
		super(RNN, self).__init__()
		

		self.LSTM1 = tf.keras.layers.LSTM(units=50,return_sequences=True, input_shape=(config.NUMBARS, 5))
		self.LSTM2 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM3 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM4 = tf.keras.layers.LSTM(units=50,)
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.dense1 = tf.keras.layers.Dense(50, activation='sigmoid')
		self.dense2 = tf.keras.layers.Dense(25, activation='sigmoid')
		self.dense3 = tf.keras.layers.Dense(1)


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
		x = self.dense3(x)
		return x

	def train(self, input, truth):
		with tf.GradientTape() as tape:
			pred = self.call(input)
			loss = config.mse(truth, pred)
		grads = tape.gradient(loss, self.trainable_variables)
		config.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return loss
