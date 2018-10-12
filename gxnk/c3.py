# -*- coding:utf-8 -*-
import gym


def policy_evaluate(grid_mdp, pi, v):
    for i in range(1000):
        delta = 0.0
        for state in grid_mdp.states:
            if state in grid_mdp.terminate_states:
                continue
            action = pi[state]
            s, r, t, _ = grid_mdp.transform(state, action)
            new_v = r + grid_mdp.gamma * v[s]
            delta = max(delta, abs(v[state] - new_v))
            v[state] = new_v
        if delta < 1e-6:
            break


def policy_improve(grid_mdp, pi, v):
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


def policy_iterate(grid_mdp):
    pi = dict()
    v = dict()
    for state in grid_mdp.states:
        if state in grid_mdp.terminate_states:
            pi[state] = None
            v[state] = 0
        else:
            pi[state] = grid_mdp.actions[0]
            v[state] = 0

    for i in range(100):
        policy_evaluate(grid_mdp, pi, v)
        policy_improve(grid_mdp, pi, v)
    return pi, v


def value_iteration(grid_mdp):
    pi = dict()
    v = dict()
    for state in grid_mdp.states:
        if state in grid_mdp.terminate_states:
            pi[state] = None
            v[state] = 0
        else:
            pi[state] = grid_mdp.actions[0]
            v[state] = 0

    for i in range(1000):
        delta = 0.0
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
            delta = max(delta, abs(v[state] - v1))
            pi[state] = a1
            v[state] = v1
        if delta < 1e-6:
            break
    return pi, v


if __name__ == '__main__':
    env = gym.make('GridWorld-v0')

    pi_policy, v_policy = policy_iterate(env.env)
    print('best policy by policy iterate: %s' % pi_policy)

    pi_value, v_value = value_iteration(env.env)
    print('best policy by value iterate: %s' % pi_value)

    env.reset()
    env.render()

    signal = input("waiting for close...")
    print('signal: %s' % signal)

    env.close()
