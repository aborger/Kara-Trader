from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])

class DQN(tf.keras.model):
	def __init__(self):
		self.LSTM1 = tf.keras.layers.LSTM(units=50,return_sequences=True, input_shape=(NUMBARS, 5))
		self.LSTM2 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM3 = tf.keras.layers.LSTM(units=50,return_sequences=True)
		self.LSTM4 = tf.keras.layers.LSTM(units=50,)
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.dense = tf.keras.layers.Dense(len(self.env.action_space))
		
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
		self.Q = defaultdict(float)
		self.gamma = 0.99

		self.env = Environment(act_buy, act_sell, act_wait, observe, reward, reset)
		self.actions = self.env.action_space


	@tf.function
	def train_step(self, tr_list):
		'''Perform training on batch of data from replay buffer'''
		# Calculate targets
		next_qs = target_nn(tr_list.s_next)
		max_next_qs = tf.reduce_max(next_qs, axis=-1)
		target = tr_list.r + (1. - tr_list.done) * discount * max_next_qs
		with tf.GradientTape() as tape:
			qs = main_nn(tr_list.s)
			action_masks = tf.one_hot(tr_list.a, len(tr_list.a))
			masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
			loss = mse(target, masked_qs)
		grads = tape.gradient(loss, main_nn.trainable_variables)
		optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
		return loss


	def act(self, state, eps=0.1):
		# Picks best action, epsilon forces creativity
		
		if np.random.rand() < eps:
			print('random action')
			return self.env.random_action()

		# Otherwise pick action with highest Q value
		qvals = {a: self.Q[state, a] for a in self.actions}
		max_q = max(qvals.values())

		# Incase multiple actions have same max Q
		actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
		return np.random.choice(actions_with_max_q)

	
	def select_epsilon_greedy_action(self, state, epsilon):
		# Epsilon is probability of random action other wise take best action
		result = tf.random.uniform((1,))
		if result < epsilon:
			return self.env.random_action()
		else:
			return tf.argmax(main_nn(state)[0]).numpy()
		
	def train(self, config):
		reward_history = []
		reward_averaged = []
		step = 0
		alpha = config.alpha
		eps = config.epsilon

		for n_episode in range(config.n_episodes):
			ob = self.env.reset()
			done = False
			reward = 0

			warmup_episodes = config.warmup_episodes
			eps_drop = (config.epsilon - config.epsilon_final) / warmup_episodes
			while not done:
				a = self.act(ob, eps)
				new_ob, r, done, info = self.env.step(a)
				
				self._update_q_value(Transition(ob, a, r, new_ob, done), alpha)

				step += 1
				reward += r
				ob = new_ob
				
				reward_history.append(reward)
				reward_averaged.append(np.average(reward_history[-50:]))
				
				alpha *= config.alpha_decay
				if eps > config.epsilon_final:
					eps = max(config.epsilon_final, eps - eps_drop)
					
				print("[episode:{}|step:{}] best:{} avg:{:.4f} alpha:{:.4f} eps:{:.4f} Qsize:{}".format(
				n_episode, step, np.max(reward_history),
				np.mean(reward_history[-10:]), alpha, eps, len(self.Q)))

class Environment:
	def __init__(self, act_buy, act_sell, act_wait, observe, reward, reset):
		self.action_space = [Action('Buy', act_buy), Action('Sell', act_sell), Action('Wait', act_wait)]
		self.observe_func = observe
		self.reward_func = reward
		self.reset_funct = reset
		self.count = 10
	

	def reset(self):
		self.reset_funct()
		initial_observation = self.observe_func()
		return initial_observation

	def step(self, action):
		action.perform()
		new_observation = self.observe_func()
		reward = self.reward_func()
		
		done = False
		self.count += 1
		if self.count > 10:
			done = True
		info = 0
		return new_observation, reward, done, info
		

	def random_action(self):
		return np.random.choice(self.action_space)

class Action:
	def __init__(self, name, action):
		self.perform = action

class ReplayBuffer(object):
	def __init__(self, size):
		self.buffer = deque(maxlen = size)
		
	def add(self, tr)
		self.buffer.append(tr)
		
	def __len__(self):
		return len(self.buffer)
		
	def sample(self, num_samples):
		states, actions, rewards, next_states, done = [], [], [], [], []
		idx = np.random.choice(len(self.buffer), num_samples)
		for i in idx:
			elem = self.buffer[i]
			states.append(np.array(elem.s, copy=False))
			actions.append(np.array(elem.a, copy=False))
			rewards.append(elem.r)
			next_states.append(np.array(elem.s_next, copy = False))
			dones.append(elem.done)
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards, dtype=np.float32)
		next_states = np.array(next_states)
		dones = np.array(dones, dtype=np.float32)
		tr = Transition(states, actions, rewards, next_states, dones)
		return tr


		







