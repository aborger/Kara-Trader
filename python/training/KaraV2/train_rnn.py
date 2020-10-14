from collections import defaultdict
import numpy as np

class DQN:
	def __init__(self):
		self.Q = defaultdict(float)
		self.gamma = 0.99
		self.alpha = 0.5
		self.epsilon = 1.0
		self.n_episodes = 1000

		self.env = Environment()
		self.actions = range(self.env.action_space)

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



	def act(self, state, eps=0.1):
		# Picks best action, epsilon forces creativity
		
		if np.random.rand() < eps:
			return self.env.random_action()

		# Otherwise pick action with highest Q value
		qvals = {a: self.Q[state, a] for a in self.actions}
		max_q = max(qvals.values())

		# Incase multiple actions have same max Q
		actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
		return np.random.choice(actions_with_max_q)

	
	def _update_q_value(self, tr, alpha):
		max_q_next = max([self.Q[tr.s_next, a] for a in self.actions])
		self.Q[tr.s, tr.a] += alpha * (tr.r + self.gamma * max_q_next * (1.0 - tr.done) - self.Q[tr.s, tr.a])

	def train(self):
		reward_history = []
		reward_averaged = []
		step = 0
		alpha = self.alpha
		eps = self.epsilon

		for n_episode in range(self.n_episodes):
			ob = self.env.reset()
			done = False
			reward = 0

			while not done:
				a = self.act(ob, eps)
				new_ob, r, done, info = self.env.step(a)
				
				self._update_q_value(Transition(ob, a, r, new_ob, done), alpha)

				step += 1
				reward += r

class Environment:
	def __init__(self):
		self.action_space = [Action('Buy'), Action('Sell'), Action('Wait')]
		self.observation_space = 0

	def reset(self):
		return initial_observation

	def step(self, action):
		

		return new_observation, reward, done, info

	def random_action(self):
		return np.random.choice(self.action_space)

class Action:
	def __init__(self, action):
		self.action = action









