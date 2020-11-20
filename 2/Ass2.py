import numpy as np
import gym
import random

alpha = 0.1
max_steps = 250
# episodes = 100000
episodes = 10000
gamma = 0.95
epsilon = 0.1
_lambda = 0
NUMBER_OF_EVAL_SIMS = 15


def init_Q(env):
    Q = np.zeros((env.env.nS, env.env.nA))
    return Q


def init_E(env):
    E = np.zeros((env.env.nS, env.env.nA))
    return E


def eps_greedy_policy(state, Q, env):
    if random.uniform(0, 1) <= epsilon:
        new_action = env.action_space.sample()  # explore
    else:
        new_action = np.argmax(Q[state, :])  # exploit
    return new_action


def sarsa_lambda(env, episodes=episodes, max_steps=max_steps):
    Q = init_Q(env)
    total_steps = 0

    for k in range(episodes):

        # init E,S,A
        E = init_E(env)
        state = env.reset()
        action = eps_greedy_policy(state, Q, env)

        actual_steps = -1
        for step in range(max_steps):
            # Take action A, observe R,S'
            new_state, reward, done, _ = env.step(action)
            new_action = eps_greedy_policy(new_state, Q, env)

            delta_error = reward + gamma * Q[new_state, new_action] - Q[state, action]
            E[state, action] += 1
            Q = np.add(Q, np.multiply(alpha * delta_error, E))
            E = np.multiply(gamma * _lambda, E)

            state = new_state
            action = new_action
            if done:
                # decay epsilon and alpha each episode (maybe will improve results)
                # tune_params()
                actual_steps = step + 1
                break

        total_steps += actual_steps if actual_steps != -1 else max_steps
        # Do here something every number of steps:
        # if total_steps%< == 0:

    return Q, np.argmax(Q, axis=1)  # returns Q and the policy


def decode_action(action):
    if action == 0:
        return "L"
    if action == 1:
        return "D"
    if action == 2:
        return "R"
    if action == 3:
        return "U"
    else:
        return "Unknown"


def policy_eval(policy, env):
    """
    policy should be an iterable with length of number of states (action per state)
    """
    rewards = []
    for i in range(NUMBER_OF_EVAL_SIMS):
        state = env.reset()

        run_reward = 0
        is_done = False
        while not is_done:
            state, reward, is_done, _ = env.step(policy[state])

            run_reward += reward

        rewards.append(run_reward)

    return np.mean(rewards)


def run_on_4_4_lake():
    env = gym.make('FrozenLake-v0')
    Q, Policy = sarsa_lambda(env, episodes, max_steps)
    env.reset()
    env.render()
    print(np.array([decode_action(action) for action in Policy]).reshape((4, 4)))
    print(policy_eval(Policy, env))


def run_on_8_8_lake():
    env = gym.make('FrozenLake8x8-v0')
    Q, Policy = sarsa_lambda(env, episodes, max_steps)
    env.reset()
    env.render()
    print(np.array([decode_action(action) for action in Policy]).reshape((8, 8)))
    print(policy_eval(Policy, env))


if __name__ == '__main__':
    run_on_4_4_lake()