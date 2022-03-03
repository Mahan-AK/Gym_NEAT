#!/anaconda3/bin/python
import gym
import numpy as np

def mod_len(o):
    try:
        return len(o)
    except:
        return 1

env_name = "LunarLander-v2"
env = gym.make(env_name)

observation = env.reset()

print(f"Envoriment {env_name}:")
print(f"Init vector: {observation} , dim = {mod_len(observation)}")
print(f"Action space: {env.action_space}, dim = {mod_len(env.action_space.sample())} ({env.action_space.sample()})")
print("\nRendering simulation with random actions...")

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(f"action taken: {reward}", end='\r')

    env.render()

print()
print(reward)