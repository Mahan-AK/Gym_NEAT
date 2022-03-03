#!/anaconda3/bin/python
import os
import gym
import neat
import numpy as np
import pickle

with open('winner_MountainCar', 'rb') as f:
    model = pickle.load(f)

config_path = "config_MountainCar"

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, config_path)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)


net = neat.nn.FeedForwardNetwork.create(model, config)

env_name = "MountainCar-v0"
env = gym.make(env_name)

observation = env.reset()

done = False
while not done:
    action = np.argmax(net.activate(observation))

    observation, reward, done, info = env.step(action)
    print(f"action taken: {action}", end='\r')

    env.render()

print()