import gym
import time

env = gym.make('CartPole-v0')
env.reset()
env.render()

time.sleep(3)

env.close()
