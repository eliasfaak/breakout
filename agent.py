import os
os.environ["KERAS_BACKEND"]="plaidml.keras.backend"
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
import gym
import numpy as np


class ReplayBuffer(object):
	def __init__(self, max_size, input_shape, n_actions, discrete=False):
		self.mem_counter = 0
		self.mem_size = max_size
		self.input_shape = input_shape
		self.discrete = discrete
		dtype = np.int8 if self.discrete else np.float32
		self.state_memory = np.zeros((self.mem_size, input_shape))
		self.new_state_memory = np.zeros((self.mem_size, input_shape))
		self.action_memory = np.zeros((self.mem_size, input_shape),dtype= dtype)
		self.reward_memory = np.zeros(self.mem_size)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)


	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_counter % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.terminal_memory[index] = 1-int(done)
		if self.discrete:
			actions = np.zeros(self.action_memory.shape[1])
			actions[action] = 1.0
			self.action_memory[index] = actions
		else:
			self.action_memory[index] = action
			self.mem_counter+=1


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
	model = Sequential([
	    Dense(fc1_dims, input_shape=(input_dims, )),
	    Activation("relu"),
	    Dense(fc2_dims),
	    Activation("relu"),
	    Dense(n_actions)])
	model.compile(optimizer=Adam(lr=lr), loss="mse")
	return model

class Agent(object):
	def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
				input_dims, epsilon_dec=0.001, epsilon_end=0.01,
				mem_size=1000000):

		self.action_space = [i for i in range(n_actions)]
		self.n_actions = n_actions
		self.gamma = gamma
		self.epsilon = epsilon
		self.alpha = alpha
		self.epsilon_dec = epsilon_dec
		self.epsilon_min = epsilon_end
		self.batch_size = batch_size

		self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
		discrete = True)
		self.q_eval = build_dqn(alpha, n_actions, input_dims, 50,50)


	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def choose_action(self, state):
		state = state[np.newaxis, :]
		rand = np.random.random()
		if rand < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			actions = self.q_eval.predict(state)
			action = np.argmax(actions)

		return action

	def learn(self):
		if self.memory.mem_counter < self.batch_size:
			return

		state, action, reward, new_state, done = \
		self.memory.sample_buffer(self.batch_size)
		action_values = np.array(self.action_space, dtype=np.int8)
		action_indices = np.dot(action, action_values)

		q_eval = self.q_eval.predict(state)
		q_next = self.q_eval.predict(new_state)

		q_target = q_eval.copy()

		batch_index = np.arange(self.batch_size, dtype=np.int32)

		q_target[batch_index, action_indices]= reward + \
		self.gamma*np.max(q_next, axis=1)*done

		_ = self.q_eval.fit(state, q_target, verbose=0)

		self.epsilon = self.epsilon_min*self.epsilon_dec if self.epsilon > \
		self.epsilon_min else self.epsilon_min


	def save_model(self):
		self.q_eval.save("breakout.h5")

	def load_model(self):
		self.q_eval = load_model("breakout.h5")
