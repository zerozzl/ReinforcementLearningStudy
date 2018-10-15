# -*- coding:utf-8 -*-
import gym

if __name__ == '__main__':
    env = gym.make('Breakout-v0')

    env.reset()
    env.render()
    signal = input("waiting for close...")
    print('signal: %s' % signal)

    env.close()
