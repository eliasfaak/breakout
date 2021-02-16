import os
from agent import Agent
os.environ["KERAS_BACKEND"]="plaidml.keras.backend"
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
import gym
import numpy as np
if __name__ == "__main__":
    env = gym.make("Breakout-ram-v0")
    player = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=9,
                    n_actions = 4, mem_size=1000000, batch_size=64, epsilon_end=0.01)
    player.load_model()
    def train_n_games(n):
        game_counter = 0
        for i in range(n):
            obs = env.reset()
            done = False
            while not done:
                action = player.choose_action(obs)
                obs, reward, done, info = env.step(action)
                player.learn()
            game_counter+=1
            player.save_model()
            print("games played = %i with reward %f" % (game_counter, reward))

    def play_n_games(n):
        for i in range(n):
            obs = env.reset()
            done = False
            while not done:
                env.render()
                action = player.choose_action(obs)
                obs, reward, done, info = env.step(action)
    #train_n_games(100)
    play_n_games(2)
