from collections import deque
import numpy as np
import tensorflow as tf

class config:
	num_episodes = 1000
	epsilon = 1.0
	batch_size = 32
	discount = 0.99
	NUMBARS = 10
	NUMTRADES = 30
class DQN(tf.keras.Model):
	def __init__(self):
		super(DQN, self).__init__()
		self.LSTM1 = tf.keras.layers.LSTM(units=50,return_sequences=True, input_shape=(config.NUMBARS, 5))
		self.LSTM2 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM3 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM4 = tf.keras.layers.LSTM(units=50,)
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.dense = tf.keras.layers.Dense(3)
		
	def create_model(self):
		model = Sequential()
		state_shape = self.env.observation_space.shape
		
		model.add(LSTM(units=50,return_sequences=True, input_shape=(NUMBARS, 5)))
		model.add(Dropout(0.2))

		model.add(LSTM(units=50,return_sequences=True))
		model.add(Dropout(0.2))

		model.add(LSTM(units=50,return_sequences=True))
		model.add(Dropout(0.2))

		model.add(LSTM(units=50,))
		model.add(Dropout(0.2))

		model.add(Dense(self.env.action_space.n))

		model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

		return model
		
	def call(self, x):
		# Pass forward
		x = self.LSTM1(x)
		x = self.dropout(x)
		x = self.LSTM2(x)
		x = self.dropout(x)
		x = self.LSTM3(x)
		x = self.dropout(x)
		x = self.LSTM4(x)
		x = self.dropout(x)
		return self.dense(x)


class Main:
	def __init__(self, act_buy, act_sell, act_wait, observe, reward, reset):
		self.env = Environment(act_buy, act_sell, act_wait, observe, reward, reset)
		self.config = config
		self.buffer = ReplayBuffer(100000)
		self.cur_frame = 0
		self.main_nn = DQN()
		self.target_nn = DQN()
		
		self.optimizer = tf.keras.optimizers.Adam(1e-4)
		self.mse = tf.keras.losses.MeanSquaredError()


	def train_step(self, state, action, reward, next_state, done):
		'''Perform training on batch of data from replay buffer'''
		# Calculate targets
		next_qs = self.target_nn(next_state)
		max_next_qs = tf.reduce_max(next_qs, axis=-1)
		target = reward + (1. - done) * config.discount * max_next_qs
		with tf.GradientTape() as tape:
			qs = self.main_nn(state)
			action_mask = tf.one_hot(action, len(self.env.action_space))
			masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
			loss = self.mse(target, masked_qs)
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
		
	def train(self):
		last_100_ep_rewards = []
		best = {
			"equity": 0,
			"model": 0,
			"actions": 0
		}
		for episode in range(config.num_episodes+1):
			state = self.env.reset()
			ep_reward, done = 0, False
			action_list = []
			while not done:
				#state_in = tf.expand_dims(state, axis=0)
				action = self.select_epsilon_greedy_action(state, config.epsilon)
				action_list.append(self.env.action_space[action].name)
				next_state, reward, done, info = self.env.step(action)
				ep_reward += reward
				# Save to experience replay
				self.buffer.add(state, action, reward, next_state, done)
				state = next_state
				self.cur_frame += 1
				# copy main_nn weights to target_nn
				if self.cur_frame % 2000 == 0:
					self.target_nn.set_weights(self.main_nn.get_weights())
				
				# Train neural network
				if len(self.buffer) >= config.batch_size:
					state, action, reward, next_state, dones= self.buffer.sample()
					loss = self.train_step(state, action, reward, next_state, done)
				
			if episode < 950:
				config.epsilon -= 0.001
				
			if len(last_100_ep_rewards) == 100:
				last_100_ep_rewards = last_100_ep_rewards[1:]
			last_100_ep_rewards.append(ep_reward)
			
			if ep_reward > best["equity"]:
				best["equity"] = ep_reward
				best["model"] = self.target_nn
				best["actions"] = action_list
			
			if episode % 50 == 0:
				print(f'Episode {episode}/{config.num_episodes}. Epsilon: {config.epsilon:.3f}. '
				f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
		
		print('Best Equity: ' + str(best["equity"]))
		print('Actions:')
		print(best["actions"])
		print('Saving model as Kara_V2_Model')
		best["model"].save('data/models/Kara_V2_Model.h5')
class Environment:
	def __init__(self, act_buy, act_sell, act_wait, observe, reward, reset):
		self.action_space = [Action('Buy', act_buy), Action('Sell', act_sell), Action('Wait', act_wait)]
		self.observe_func = observe
		self.reward_func = reward
		self.reset_funct = reset
		self.count = 10
	

	def reset(self):
		self.count = 0
		self.reset_funct()
		initial_observation = self.observe_func()
		return initial_observation

	def step(self, action):
		self.action_space[action].perform()
		new_observation = self.observe_func()
		reward = self.reward_func()
		done = False
		self.count += 1
		if self.count > config.NUMTRADES:
			done = True
		info = 0
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
		#states.append(np.array(state, copy=False))
		#actions.append(np.array(action, copy=False))
		#rewards.append(reward)
		#next_states.append(np.array(next_state, copy = False))
		#dones.append(done)
		#states = np.array(states)
		#actions = np.array(actions)
		#rewards = np.array(rewards, dtype=np.float32)
		#next_states = np.array(next_states)
		#dones = np.array(dones, dtype=np.float32)
		return state, action, reward, next_state, done


		







