from collections import deque
import numpy as np
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import multiprocessing

class config:
	num_episodes = 500
	epsilon = 1
	epsilon_reset = 1
	epsilon_discount = 0.95
	batch_size = 32
	discount = 0.99
	NUMBARS = 10
	NUMTRADES = 30
	NUMSTOCKS = 10

	NUM_ACTIONS = 3

	optimizer = tf.keras.optimizers.Adam(1e-4)
	mse = tf.keras.losses.MeanSquaredError()


def checkError(val):
	if val.numpy().any() < -1 or val.numpy().any() > 1:
		print(val)
		raise ValueError(val)

def feed_EDQN(EDQN, total_state):
	# total_state[0] = stock_bars[NUMSTOCKS, NUMBARS, 5]. total_state[1] = position_size[NUMSTOCKS]. total_state[2] = buying_power_ratio[NUMSTOCKS]
	# get eval for each stock from EDQN
	stock_bars = total_state[0]
	evals = []
	for stock in range(0, config.NUMSTOCKS):
		eval = EDQN(stock_bars[stock])
		evals.append(eval)

	# reshape to feed to CDQN
	evals = np.array(evals)
	evals = evals.reshape((config.NUMSTOCKS))

	# normalize position size
	position_size = np.array(total_state[1])
	position_size = position_size.reshape(-1, 1)
	sc = MinMaxScaler()
	position_size = sc.fit_transform(position_size)
	position_size = position_size.reshape(1 , -1)

	# normalize buying power ratio
	buying_power_ratio = np.array(total_state[2])
	buying_power_ratio = buying_power_ratio.reshape(-1, 1)
	sc = MinMaxScaler()
	buying_power_ratio = sc.fit_transform(buying_power_ratio)
	buying_power_ratio = buying_power_ratio.reshape(1, -1)

	mid_state = np.vstack((evals, position_size, buying_power_ratio))
	return mid_state

def format_CDQN_output(output):
	# format output
	max_action = tf.argmax(output, axis=1).numpy()
	best_action = -1
	best_stock = -1
	best_value = -100
	for action in range(0, len(max_action)):
		stock = tf.argmax(output[:, max_action[action]]).numpy()
		value = output[stock, action]
		if value > best_value:
			best_value = value
			best_action = action
			best_stock = stock

	return (best_action, best_stock)

# -------------------------------- Models ----------------------------------------------
# Each stock goes through this EDQN model to be evaluated
# NUMBARS worth of bar data gets converted into a single evaluation value
# EDQN (Evaluation Deep Q-Network)
tf.keras.backend.set_floatx('float64')
class EDQN(tf.keras.Model):
	def __init__(self):
		super(EDQN, self).__init__()
		self.LSTM1 = tf.keras.layers.LSTM(units=50,return_sequences=True, input_shape=(config.NUMBARS, 5))
		self.LSTM2 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM3 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM4 = tf.keras.layers.LSTM(units=50,)
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.dense1 = tf.keras.layers.Dense(50)
		self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

		self.reward_size = 0
		self.reward_mean = 0
		self.reward_variance = 1


	def call(self, input):
		# normalize
		input = np.array(input)
		sc = MinMaxScaler()
		input = sc.fit_transform(input)
		input = np.reshape(input, (1, config.NUMBARS, 5))

		# Pass forward
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
		return x

	def train(self, state, reward):
		self.reward_size += 1
		new_mean = self.reward_mean + (reward - self.reward_mean)/self.reward_size
		new_variance = (self.reward_variance + (reward - self.reward_mean) * (reward - new_mean)) / self.reward_size
		self.reward_mean = new_mean
		self.reward_variance = new_variance
		target_score = (reward - self.reward_mean) / math.sqrt(self.reward_variance)

		'''Perform training on batch of data from replay buffer'''
		# Calculate targets

		with tf.GradientTape() as tape:
			q = self.call(state)
			loss = config.mse(target_score, q)
		grads = tape.gradient(loss, self.trainable_variables)
		config.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return loss

# Each user has a CDQN, input is and (evaluation, position size, buying power ratio) for each stock
# outputs a (buy/sell/wait) for each stock
# CDQN (Chooser Deep Q-Network), I couldnt think of anything better than chooser, but this one picks which stock
class CDQN(tf.keras.Model):
	def __init__(self):
		super(CDQN, self).__init__()
		self.dense1 = tf.keras.layers.Dense(config.NUMSTOCKS, input_shape=(3, config.NUMSTOCKS), activation='relu')
		self.dense2 = tf.keras.layers.Dense(config.NUMSTOCKS)
		self.dense3 = tf.keras.layers.Dense(config.NUMSTOCKS, activation='sigmoid')




	def call(self, input):
		x = input
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dense3(x)
		checkError(x)
		return x


	
	def train(self, target_nn, env, state, action, reward, next_state, done):
		'''Perform training on batch of data from replay buffer'''
		# Calculate targets
		next_qs = target_nn(next_state)
		max_next_qs = tf.reduce_max(next_qs, axis=-1)
		target = reward + (1. - done) * config.discount * max_next_qs
		with tf.GradientTape() as tape:
			qs = self.call(state)
			# Create one hot
			action_mask = tf.zeros([3, config.NUMSTOCKS]).numpy()
			action_mask[action] = 1
			masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
			loss = config.mse(target, masked_qs)
		grads = tape.gradient(loss, self.trainable_variables)
		config.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return loss



class Main:
	def __init__(self, act_buy, act_sell, act_wait, observe, reward, reset):
		self.env = Environment(act_buy, act_sell, act_wait, observe, reward, reset)
		config.epsilon = config.epsilon_reset
		self.buffer = ReplayBuffer(100000)
		self.EvalBuffer = ReplayBuffer(100000)
		self.ChooseBuffer = ReplayBuffer(100000)
		self.cur_frame = 0
		self.target_EDQN = EDQN()
		self.main_CDQN = CDQN()
		self.target_CDQN = CDQN()

	# total_state[0] = stock_data[NUMSTOCKS, NUMBARS, 5]. total_state[1] = position size[NUMSTOCKS]. total_state[2] = buying power[1]
	def select_epsilon_greedy_action(self, total_state, epsilon):
		# Epsilon is probability of random action other wise take best action
		result = tf.random.uniform((1,))
		if result < epsilon:
			return self.env.random_action(), None
		else:

			mid_state = feed_EDQN(self.target_EDQN, total_state)
			
			# feed to CDQN
			output = self.main_CDQN(mid_state)
			return format_CDQN_output(output), output
			
		
	def continue_training(self, model):
		config.epsilon = .5
		config.epsilon_discount = .45
		self.main_nn.set_weights(model.get_weights())
		model = self.train()
		return model

		
	def train(self):
		last_100_ep_equities = []
		last_100_ep_rewards = []
		last_100_ep_buying_errors = []
		last_100_ep_selling_errors = []
		best = {
			"equity": 0,
			"reward": 0,
			"model": 0,
			"actions": 0,
			"output": None
		}
		for episode in tqdm(range(config.num_episodes+1)):
			total_state = self.env.reset()
			ep_equity, ep_reward, done, buying_errors, selling_errors = 0, 0, False, 0, 0
			action_list = []
			output = 0
			while not done:
				action, output = self.select_epsilon_greedy_action(total_state, config.epsilon)
				action_list.append(action)
				next_total_state, total_reward, done, info = self.env.step(action)
				ep_equity = total_reward[1]
				buying_errors = info[0]
				selling_errors = info[1]
				try:
					ep_reward = ep_equity / (buying_errors + selling_errors)
				except ZeroDivisionError:
					ep_reward = ep_equity * 1.5
				
				new_reward = (total_reward[0], ep_reward)
				# Save to experience replay
				self.buffer.add(total_state, action, new_reward, next_total_state, done)

				total_state = next_total_state
				self.cur_frame += 1
				# copy main_nn weights to target_nn
				if self.cur_frame % 2000 == 0:
					self.target_CDQN.set_weights(self.main_CDQN.get_weights())
				
				# Train neural networks
				if len(self.buffer) >= config.batch_size:
					total_state, action, total_reward, next_total_state, dones = self.buffer.sample()

					# Get mid_states for CDQN
					mid_state = feed_EDQN(self.target_EDQN, total_state)
					next_mid_state = feed_EDQN(self.target_EDQN, next_total_state)

					# train EDQN
					stock_bars = total_state[0]
					for stock in range(0, config.NUMSTOCKS):
						self.target_EDQN.train(stock_bars[stock], total_reward[0][stock])

					# train CDQN
					loss = self.main_CDQN.train(self.target_CDQN, self.env, mid_state, action, total_reward[1], next_mid_state, dones)
					
				
			if episode < config.num_episodes * config.epsilon_discount:
				config.epsilon -= config.epsilon_discount / config.num_episodes
				
			if len(last_100_ep_rewards) == 100:
				last_100_ep_equities = last_100_ep_equities[1:]
				last_100_ep_rewards = last_100_ep_rewards[1:]
				last_100_ep_buying_errors = last_100_ep_buying_errors[1:]
				last_100_ep_selling_errors = last_100_ep_selling_errors[1:]

			last_100_ep_equities.append(ep_equity)
			last_100_ep_rewards.append(ep_reward)
			last_100_ep_buying_errors.append(buying_errors)
			last_100_ep_selling_errors.append(selling_errors)
			

			if ep_reward > best["reward"]:
				best["equity"] = ep_equity
				best["reward"] = ep_reward
				best["model"] = (self.target_EDQN, self.target_CDQN)
				best["actions"] = action_list
				if output is not None:
					best["output"] == output
			# allows there to be an output as soon as the models are useds from select_epsilon
			if output is not None:
				best["output"] = output
			
			if episode % 10 == 0:
				print(f'Episode {episode}/{config.num_episodes}. Epsilon: {config.epsilon:.3f}. \n'
				f'Last 10 episodes: \n'
				f'Average Equity: {np.mean(last_100_ep_equities):.2f}\n'
				f'Average Reward: {np.mean(last_100_ep_rewards):.3f}\n'
				f'Average Buying Error: {np.mean(last_100_ep_buying_errors):.1f}\n'
				f'Average Selling Error: {np.mean(last_100_ep_selling_errors):.1f}\n'
				f'CDQN Output: {best["output"]}\n')
		
		print('Best Equity: ' + str(best["equity"]))
		print('Best Reward: ' + str(best["reward"]))
		print('Actions:')
		print(best["actions"])
		print('Saving model as Kara_V2_Model')
		return best

class Environment:
	def __init__(self, act_buy, act_sell, act_wait, observe, reward, reset):
		self.action_space = [Action('Buy', act_buy), Action('Sell', act_sell), Action('Wait', act_wait)]
		self.observe_func = observe
		self.reward_func = reward
		self.reset_funct = reset
		self.count = 0
		self.buying_errors = 0
		self.selling_errors = 0

	def reset(self):
		self.count = 0
		self.buying_errors = 0
		self.selling_errors = 0
		self.reset_funct()
		initial_observation = self.observe_func()
		return initial_observation

	def step(self, action):
		success = self.action_space[action[0]].perform(action[1])
		total_state = self.observe_func()
		total_reward = self.reward_func()

		if not success:
			if (self.action_space[action[0]].name == 'Buy'):
				self.buying_errors += 1
			elif (self.action_space[action[0]].name == 'Sell'):
				self.selling_errors += 1
		#print('Reward: ' + str(reward))
		done = False
		self.count += 1
		if self.count > config.NUMTRADES:
			done = True
		info = (self.buying_errors, self.selling_errors)
		return total_state, total_reward, done, info
		

	def random_action(self):
		action = np.random.randint(0, len(self.action_space))
		stock = np.random.randint(0, config.NUMSTOCKS)
		return (action, stock)

class Action:
	def __init__(self, name, action):
		self.perform = action
		self.name = name

class ReplayBuffer(object):
	def __init__(self, size):
		self.buffer = deque(maxlen = size)
		
	def add(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))
		
	def __len__(self):
		return len(self.buffer)
		
	def sample(self):
		states, actions, rewards, next_states, dones = [], [], [], [], []
		idx = np.random.choice(len(self.buffer))

		elem = self.buffer[idx]
		state, action, reward, next_state, done = elem

		return state, action, reward, next_state, done


		







