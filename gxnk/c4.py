# -*- coding:utf-8 -*-
import gym
import random


def mc_eval(grid_mdp, state_sample, action_sample, reward_sample):
    vfunc = dict()
    nfunc = dict()
    for s in grid_mdp.states:
        vfunc[s] = 0.0
        nfunc[s] = 0.0
    for iter1 in range(len(state_sample)):
        G = 0.0
        for step in range(len(state_sample[iter1]) - 1, -1, -1):
            G *= grid_mdp.gamma
            G += reward_sample[iter1][step]
        for step in range(len(state_sample[iter1])):
            s = state_sample[iter1][step]
            vfunc[s] += G
            nfunc[s] += 1.0
            G -= reward_sample[iter1][step]
            G /= grid_mdp.gamma
    for s in grid_mdp.states:
        if nfunc[s] > 0.000001:
            vfunc[s] /= nfunc[s]
    return vfunc


def mc(grid_mdp, num_iter1, epsilon):
    x = []
    y = []
    n = dict()
    qfunc = dict()
    for s in grid_mdp.states:
        for a in grid_mdp.actions:
            qfunc['%s_%s' % (s, a)] = 0.0
            n['%s_%s' % (s, a)] = 0.001

    for iter1 in range(num_iter1):
        x.append(iter1)
        y.append(compute_error(qfunc))
        s_sample = []
        a_sample = []
        r_sample = []
        s = grid_mdp.states[int(random.random() * len(grid_mdp.states))]
        t = False
        count = 0
        while (t is False) and (count < 100):
            a = epsilon_greedy(qfunc, s, epsilon)
            s1, r, t, _ = grid_mdp.transform(s, a)
            s_sample.append(s)
            a_sample.append(a)
            r_sample.append(r)
            s = s1
            count += 1
        g = 0.0
        for i in range(len(s_sample) - 1, -1, -1):
            g *= grid_mdp.gamma
            g += r_sample[i]
        for i in range(len(s_sample)):
            key = '%s_%s' % (s_sample[i], a_sample[i])
            n[key] += 1.0
            qfunc[key] = (qfunc[key] * (n[key] - 1) + g) / n[key]
            g -= r_sample[i]
            g /= grid_mdp.gamma
    return qfunc
