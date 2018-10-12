import gym
import time

env = gym.make('LunarLander-v2')
env.reset()
env.render()

time.sleep(3)

env.close()
