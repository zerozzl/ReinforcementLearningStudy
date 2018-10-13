# -*- coding:utf-8 -*-
import gym
import random


def policy_improve_by_v(grid_mdp, pi, v):
    for state in grid_mdp.states:
        if state in grid_mdp.terminate_states:
            continue
        a1 = grid_mdp.actions[0]
        s, r, t, _ = grid_mdp.transform(state, a1)
        v1 = r + grid_mdp.gamma * v[s]
        for action in grid_mdp.actions:
            s, r, t, _ = grid_mdp.transform(state, action)
            if v1 < r + grid_mdp.gamma * v[s]:
                a1 = action
                v1 = r + grid_mdp.gamma * v[s]
        pi[state] = a1


def mc_eval(grid_mdp, state_sample, action_sample, reward_sample):
    vfunc = dict()
    nfunc = dict()
    for s in grid_mdp.states:
        vfunc[s] = 0.0
        nfunc[s] = 0.0
    for iter in range(len(state_sample)):
        G = 0.0
        for step in range(len(state_sample[iter]) - 1, -1, -1):
            G *= grid_mdp.gamma
            G += reward_sample[iter][step]
        for step in range(len(state_sample[iter])):
            s = state_sample[iter][step]
            vfunc[s] += G
            nfunc[s] += 1.0
            G -= reward_sample[iter][step]
            G /= grid_mdp.gamma
    for s in grid_mdp.states:
        if nfunc[s] > 0.000001:
            vfunc[s] /= nfunc[s]
    return vfunc


# epsilon贪婪策略
def epsilon_greedy(qfunc, state, actions, epsilon):
    amax = 0
    key = "%d_%s" % (state, actions[0])
    qmax = qfunc[key]
    for i in range(len(actions)):  # 扫描动作空间得到最大动作值函数
        key = "%d_%s" % (state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    # 概率部分
    pro = [0.0 for _ in range(len(actions))]
    pro[amax] += 1 - epsilon
    for i in range(len(actions)):
        pro[i] += epsilon / len(actions)

    # 选择动作
    r = random.random()
    s = 0.0
    for i in range(len(actions)):
        s += pro[i]
        if s >= r:
            return actions[i]
    return actions[len(actions) - 1]


def mc(grid_mdp, num_iter, epsilon):
    n = dict()
    qfunc = dict()
    for s in grid_mdp.states:
        for a in grid_mdp.actions:
            qfunc['%s_%s' % (s, a)] = 0.0
            n['%s_%s' % (s, a)] = 0.001

    for iter1 in range(num_iter):
        s_sample = []
        a_sample = []
        r_sample = []
        s = grid_mdp.states[int(random.random() * len(grid_mdp.states))]
        t = False
        count = 0
        while (t is False) and (count < 100):
            a = epsilon_greedy(qfunc, s, grid_mdp.actions, epsilon)
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


def policy_improve_by_q(grid_mdp, pi, q):
    for state in grid_mdp.states:
        if state in grid_mdp.terminate_states:
            continue
        best_a = grid_mdp.actions[0]
        best_r = q['%s_%s' % (state, best_a)]
        for action in grid_mdp.actions:
            key = '%s_%s' % (state, action)
            r = q[key]
            if best_r < r:
                best_a = action
                best_r = r
        pi[state] = best_a


if __name__ == '__main__':
    env = gym.make('GridWorld-v0')

    state_sample, action_sample, reward_sample = env.env.generate_random_sample(100)
    v_eval = mc_eval(env.env, state_sample, action_sample, reward_sample)
    pi_1 = dict()
    policy_improve_by_v(env.env, pi_1, v_eval)
    print('best policy 1: %s' % pi_1)

    q_eval = mc(env.env, 100, 0.3)
    pi_2 = dict()
    policy_improve_by_q(env.env, pi_2, q_eval)
    print('best policy 2: %s' % pi_2)

    env.reset()
    env.render()

    signal = input("waiting for close...")
    print('signal: %s' % signal)

    env.close()
