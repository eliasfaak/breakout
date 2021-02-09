import os
os.environ["KERAS_BACKEND"]="plaidml.keras.backend"
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
import gym
import numpy as np


class ReplayBufer(object):
	def __init__(self, max_size, input_shape, n_actions, discrete=False):
		self.mem_counter = 0
		self.mem_size = max_size
		self.input_shape = input_shape
		self.discrete = discrete
		dtype = np.int8 if self.discrete else np.float32
		self.state_memory = np.zeros((mem_size, input_shape))
		self.new_state_memory = np.zeros((mem_size, input_shape))
		self.action_memory = np.zeros((mem_size, input_shape),dtype= dtype)
		self.reward_memory = np.zeros(mem_size)
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
        self.mem_cntr+=1



class Agent():
	def __int__(self, alpha, gamma, n_actions, epsilon, batch_size,
                    input_dims,fname, epsilon_dec=0.001, epsilon_end=0.01,
                    mem_size=1000000):

        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
