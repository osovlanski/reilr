import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from itertools import product


alpha = 0.05
max_steps = 250
episodes = 1000000
gamma = 0.95
epsilon_max = 1.0
epsilon_min = 0.1
_lambda = 0.6
NUMBER_OF_EVAL_SIMS = 200
EVAL_WITH_DISCOUNT = False


def init_Q(env):
    Q = np.zeros((env.env.nS, env.env.nA))
    return Q


def init_E(env):
    E = np.zeros((env.env.nS, env.env.nA))
    return E


def eps_greedy_policy(epsilon, state, Q, env):
    if random.uniform(0, 1) < epsilon:
        new_action = env.action_space.sample()  # explore
    else:
        new_action = np.argmax(Q[state, :])  # exploit
    return new_action


def sarsa_lambda(env, episodes=episodes, max_steps=max_steps,
                 epsilon_max=epsilon_max, epsilon_min=epsilon_min, is_decay=True, _lambda=_lambda, alpha=alpha):

    Q = init_Q(env)
    total_steps = 0
    epsilon = epsilon_max
    policy_vals = []

    for k in range(episodes):
        
        # init E,S,A
        E = init_E(env)
        state = env.reset()
        action = eps_greedy_policy(epsilon, state, Q, env)

        for step in range(max_steps):
            # Take action A, obvserve R,S'
            new_state, reward, done, _ = env.step(action)
            new_action = eps_greedy_policy(epsilon, new_state, Q, env)
            
            delta_error = reward + gamma * Q[new_state, new_action] - Q[state, action]
            E[state, action] += 1
            Q = np.add(Q, np.multiply(alpha * delta_error, E))
            E = np.multiply(gamma * _lambda, E)
            
            state = new_state
            action = new_action
            total_steps += 1

            if (total_steps < 20000 and total_steps % 4000 == 0) or (total_steps >= 20000 and total_steps % 20000 == 0):
                policy = np.argmax(Q, axis=1)
                policy_evaluate = policy_eval(policy, env, with_discount=EVAL_WITH_DISCOUNT)
                policy_vals.append((total_steps, policy_evaluate))

            if done:
                break
        
        if is_decay:
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-0.00005 * k)

        if total_steps > 1e6:
            break

    return Q, np.argmax(Q, axis=1), policy_vals  # returns Q, the policy and the values of the policy during the run


def policy_eval(policy, env, with_discount=False):
    """
    policy should be an iterable with length of number of states (action per state)
    """
    rewards = []
    for i in range(NUMBER_OF_EVAL_SIMS):
        state = env.reset()

        run_reward = 0
        is_done = False
        steps = 0
        while not is_done:
            state, reward, is_done, _ = env.step(policy[state])
            steps += 1
            if with_discount:
                run_reward += reward * (gamma ** steps)
            else:
                run_reward += reward

        rewards.append(run_reward)

    return np.mean(rewards)


def show_sim_in_env(env, policy):
    state = env.reset()
    env.render()

    total_reward = 0
    is_done = False
    num_of_steps = 0

    while not is_done:
        action = np.argmax(policy[state])
        state, step_reward, is_done, _ = env.step(action)
        total_reward += step_reward

        num_of_steps += 1
        env.render()

    print('done in {} steps and reward: {}'.format(num_of_steps, total_reward))


def run_and_create_plot(env):
    alphas = [0.05, 0.1]
    lambdas = [0, 0.6]
    plt.figure(figsize=(12, 7))

    for alpha, _lambda in product(alphas, lambdas):
        description = 'alpha: {}, lambda: {}'.format(alpha, _lambda)
        print(description)

        Q, policy, policy_vals = sarsa_lambda(env, episodes, max_steps, alpha=alpha, _lambda=_lambda)

        show_sim_in_env(env, policy)

        plt.plot([x for x, _ in policy_vals], [y for _, y in policy_vals],
                 label=description, alpha=0.6)

    plt.xlabel('number of steps')
    plt.ylabel('avarage reward')
    plt.title('Reward For Different Hyper Parameters')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')
    run_and_create_plot(env)