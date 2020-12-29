from collections import deque
import numpy as np
import tensorflow as tf

class config:
	num_episodes = 50
	epsilon = 1.0
	epsilon_discount = 0.95
	batch_size = 32
	discount = 0.99
	NUMBARS = 10
	NUMTRADES = 10

	NUM_ACTIONS = 3

# -------------------------------- Models ----------------------------------------------
class DQN(tf.keras.Model):
	def __init__(self):
		super(DQN, self).__init__()
		self.LSTM1 = tf.keras.layers.LSTM(units=50,return_sequences=True, input_shape=(config.NUMBARS, 5))
		self.LSTM2 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM3 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM4 = tf.keras.layers.LSTM(units=50,)
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.dense_stock = tf.keras.layers.Dense(50)
		self.dense_account = tf.keras.layers.Dense(4, input_shape=(1, 2))
		self.concate = tf.keras.layers.Concatenate()
		self.dense_final = tf.keras.layers.Dense(config.NUM_ACTIONS)
		

		
	def call(self, input):
		# Pass forward
		y = self.dense_account(input[1]) # input[1] contains (buying_power, position_size)

		x = self.LSTM1(input[0]) # input[0] contains (NUMBARS, BARDATA)
		x = self.dropout(x)
		x = self.LSTM2(x)
		x = self.dropout(x)
		x = self.LSTM3(x)
		x = self.dropout(x)
		x = self.LSTM4(x)
		x = self.dropout(x)
		x = self.dense_stock(x)

		z = self.concate([x, y])
		return self.dense_final(z)


class Main:
	def __init__(self, act_buy, act_sell, act_wait, observe, reward, reset):
		self.env = Environment(act_buy, act_sell, act_wait, observe, reward, reset)
		config.epsilon = 1
		self.buffer = ReplayBuffer(100000)
		self.cur_frame = 0
		self.main_nn = DQN()
		self.target_nn = DQN()
		
		self.optimizer = tf.keras.optimizers.Adam(1e-4)
		self.mse = tf.keras.losses.MeanSquaredError()


	def train_step(self, state, action, reward, next_state, done):
		'''Perform training on batch of data from replay buffer'''
		# Calculate targets
		print()
		print()
		next_qs = self.target_nn(next_state)
		print('next_qs:')
		print(next_qs)
		max_next_qs = tf.reduce_max(next_qs, axis=-1)
		print('max_next_qs')
		print(max_next_qs)
		target = reward + (1. - done) * config.discount * max_next_qs
		print('target:')
		print(target)

		with tf.GradientTape() as tape:
			qs = self.main_nn(state)
			print('qs:')
			print(qs)
			action_mask = tf.one_hot(action, len(self.env.action_space))
			print('action_mask:')
			print(action_mask)
			masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
			print('masked_qs:')
			print(masked_qs)
			loss = self.mse(target, masked_qs)
			print('loss:')
			print(loss)
		grads = tape.gradient(loss, self.main_nn.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))
		return loss
	
	def select_epsilon_greedy_action(self, state, epsilon):
		# Epsilon is probability of random action other wise take best action
		result = tf.random.uniform((1,))
		if result < epsilon:
			return self.env.random_action()
		else:
			return tf.argmax(self.main_nn(state)[0]).numpy()
		
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
			"actions": 0
		}
		for episode in range(config.num_episodes+1):
			state = self.env.reset()
			ep_equity, ep_reward, done, buying_errors, selling_errors = 0, 0, False, 0, 0
			action_list = []
			while not done:
				#state_in = tf.expand_dims(state, axis=0)
				action = self.select_epsilon_greedy_action(state, config.epsilon)
				action_list.append(self.env.action_space[action].name)
				next_state, reward, done, info = self.env.step(action)
				ep_equity = reward
				buying_errors = info[0]
				selling_errors = info[1]
				try:
					ep_reward = ep_equity / (buying_errors + selling_errors)
				except ZeroDivisionError:
					ep_reward = ep_equity * 1.5
				
				# Save to experience replay
				self.buffer.add(state, action, ep_reward, next_state, done)
				state = next_state
				self.cur_frame += 1
				# copy main_nn weights to target_nn
				if self.cur_frame % 2000 == 0:
					self.target_nn.set_weights(self.main_nn.get_weights())
				
				# Train neural network
				if len(self.buffer) >= config.batch_size:
					state, action, reward, next_state, dones= self.buffer.sample()
					loss = self.train_step(state, action, reward, next_state, done)
				
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
				best["model"] = self.target_nn
				best["actions"] = action_list

			
			if episode % 10 == 0:
				print(f'Episode {episode}/{config.num_episodes}. Epsilon: {config.epsilon:.3f}. \n'
				f'Last 10 episodes: \n'
				f'Average Equity: {np.mean(last_100_ep_equities):.2f}\n'
				f'Average Reward: {np.mean(last_100_ep_rewards):.3f}\n'
				f'Average Buying Error: {np.mean(last_100_ep_buying_errors):.1f}\n'
				f'Average Selling Error: {np.mean(last_100_ep_selling_errors):.1f}\n')
		
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
		self.reset_funct()
		initial_observation = self.observe_func()
		return initial_observation

	def step(self, action):
		success = self.action_space[action].perform()
		new_observation = self.observe_func()
		reward = self.reward_func()

		if not success:
			if (self.action_space[action].name == 'Buy'):
				self.buying_errors += 1
			elif (self.action_space[action].name == 'Sell'):
				self.selling_errors += 1
		#print('Reward: ' + str(reward))
		done = False
		self.count += 1
		if self.count > config.NUMTRADES:
			done = True
		info = (self.buying_errors, self.selling_errors)
		return new_observation, reward, done, info
		

	def random_action(self):
		return np.random.randint(0, len(self.action_space))

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


		







