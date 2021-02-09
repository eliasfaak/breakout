import os
os.environ["KERAS_BACKEND"]="plaidml.keras.backend"
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
import gym
import numpy as np

env = gym.make("Breakout-ram-v0")
env.reset()
obs, rew, done, info = env.step(env.action_space.sample())

