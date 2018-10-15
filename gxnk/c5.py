# -*- coding:utf-8 -*-
import gym
import random


#  贪婪策略
def greedy(actions, qfunc, state):
    amax = 0
    key = "%d_%s" % (state, actions[0])
    qmax = qfunc[key]
    for i in range(len(actions)):  # 扫描动作空间得到最大动作值函数
        key = "%d_%s" % (state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    return actions[amax]


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


def sarsa(grid_mdp, num_iter, alpha, epsilon):
    qfunc = dict()  # 行为值函数为字典
    # 初始化行为值函数为0
    for s in grid_mdp.states:
        for a in grid_mdp.actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0
    for iter in range(num_iter):
        # 初始化初始状态
        s = grid_mdp.reset()
        a = grid_mdp.actions[int(random.random() * len(grid_mdp.actions))]
        t = False
        count = 0
        while (t is False) and (count < 100):
            key = "%d_%s" % (s, a)
            # 与环境进行一次交互，从环境中得到新的状态及回报
            s1, r, t1, _ = grid_mdp.step(a)
            # s1处的最大动作
            a1 = epsilon_greedy(qfunc, s1, grid_mdp.actions, epsilon)
            key1 = "%d_%s" % (s1, a1)
            # 利用qlearning方法更新值函数
            qfunc[key] = qfunc[key] + alpha * (r + grid_mdp.gamma * qfunc[key1] - qfunc[key])
            # 转到下一个状态
            s = s1;
            a = epsilon_greedy(qfunc, s1, grid_mdp.actions, epsilon)
            count += 1
    return qfunc


def qlearning(grid_mdp, num_iter, alpha, epsilon):
    qfunc = dict()  # 行为值函数为字典
    # 初始化行为值函数为0
    for s in grid_mdp.states:
        for a in grid_mdp.actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0
    for iter in range(num_iter):
        # 初始化初始状态
        s = grid_mdp.reset()
        a = grid_mdp.actions[int(random.random() * len(grid_mdp.actions))]
        t = False
        count = 0
        while (t is False) and (count < 100):
            key = "%d_%s" % (s, a)
            # 与环境进行一次交互，从环境中得到新的状态及回报
            s1, r, t1, _ = grid_mdp.step(a)
            # s1处的最大动作
            a1 = greedy(grid_mdp.actions, qfunc, s1)
            key1 = "%d_%s" % (s1, a1)
            # 利用qlearning方法更新值函数
            qfunc[key] = qfunc[key] + alpha * (r + grid_mdp.gamma * qfunc[key1] - qfunc[key])
            # 转到下一个状态
            s = s1;
            a = epsilon_greedy(qfunc, s1, grid_mdp.actions, epsilon)
            count += 1
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

    q_eval = sarsa(env.env, 100, 0.1, 0.3)
    pi_sarsa = dict()
    policy_improve_by_q(env.env, pi_sarsa, q_eval)
    print('best policy of sarsa: %s' % pi_sarsa)

    q_eval = qlearning(env.env, 100, 0.1, 0.3)
    pi_qlearning = dict()
    policy_improve_by_q(env.env, pi_qlearning, q_eval)
    print('best policy of q-learning: %s' % pi_qlearning)

    env.reset()
    env.render()

    signal = input("waiting for close...")
    print('signal: %s' % signal)

    env.close()
