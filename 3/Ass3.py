import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from itertools import product
import pickle

np.set_printoptions(precision=2)


alpha = 0.02
max_steps_in_episode = 500
total_max_steps = 10000
episodes = 10000
gamma = 1
epsilon_max = 0.1
epsilon_min = 0.005
_lambda = 0.5
NUMBER_OF_EVAL_SIMS = 100
EVAL_WITH_DISCOUNT = False
ACTION_NO = 3

BEST_WEIGHTS_PATH = 'prev_calculated_good_weights.pkl'


# -1.2 is the leftest position.
# The env starts at a location between -0.6 and -0.4 randomly.
# The agents win when he gets to 0.5
C_P = [-1.16, 0.08, 0.21, 0.38]
# -0.7 is the min speed, 0.7 is the max speed.
C_VEL = [-0.48, -0.43, -0.31, 0.03, 0.47, 0.49, 0.65, 0.66]

N = len(C_P) * len(C_VEL)

prod = product(C_P, C_VEL)
C = [np.array(val).reshape((1, -1)) for val in prod]


def init_weights():
    return np.zeros((N, ACTION_NO))


def init_E():
    return np.zeros((N, ACTION_NO))


def get_features(state):
    """
    returns a vector phi that each entry is computed feature for given p, v
    """
    # reshape to make it a matrix with one row (so we can transpose it later)
    p, v = state
    p_v = np.array([p, v]).reshape((1, -1)).T
    X = np.array([p_v - c_entry.T for c_entry in C])
    inv_cov = np.linalg.inv(np.diag([0.04, 0.0004]))
    phi = np.array([np.exp(-(xi.T @ inv_cov @ xi) / 2) for xi in X])

    return np.squeeze(phi)   # get rid of 2 unnecessary dimensions


def eps_greedy_policy(epsilon, Q):
    if random.uniform(0, 1) < epsilon:
        new_action = env.action_space.sample()  # explore
    else:
        new_action = Q.argmax()  # exploit
    return new_action


# Returns the state scaled between 0 and 1
def normalize_state(state):
    p, v = state
    p_n = (p-env.observation_space.low[0]) / (env.observation_space.high[0] - env.observation_space.low[0])
    v_n = (v-env.observation_space.low[1]) / (env.observation_space.high[1] - env.observation_space.low[1])

    return p_n, v_n


# p,v should be normalized
def get_Q(features, W):
    return np.array([features @ W[:, a] for a in range(ACTION_NO)])


def get_Q_a(features, W_a):
    return features @ W_a


def sarsa_lambda(env, episodes=episodes, max_steps=max_steps_in_episode, is_saving_weights=False,
                 epsilon_max=epsilon_max, epsilon_min=epsilon_min, is_decay=False, _lambda=_lambda, alpha=alpha):

    # (p,v,a) = (p,v,i) @ (i,a) => the shape of W should be (i,a): (32,3) 
    W = init_weights()
    total_steps = 0
    epsilon = epsilon_max
    policy_vals = []

    if is_saving_weights:
        best_policy_val = -600   # the min value of the env is -500
    
    for k in range(episodes):
        # init E,S,A
        # state = (pos,vel)
        E = init_E()
        state = normalize_state(env.reset()) 
        features = get_features(state)
        action = eps_greedy_policy(epsilon, get_Q(features, W))

        for step in range(max_steps):
            total_steps += 1

            # Take action A, obvserve R,S'
            new_state, reward, done, _ = env.step(action)
            new_state = normalize_state(new_state)
            new_features = get_features(new_state)
            new_action = eps_greedy_policy(epsilon, get_Q(new_features, W))
            curr_Q_p_v_a = get_Q_a(features, W[:, action])
            next_Q_p_v_a = get_Q_a(new_features, W[:, new_action])
            
            if done:
                delta_error = reward - curr_Q_p_v_a
            else:
                delta_error = reward + gamma * next_Q_p_v_a - curr_Q_p_v_a

            stochasticGradient = features
            E[:, action] = stochasticGradient  # replacing traces
           
            deltaW = (np.multiply(alpha*delta_error, E))
            W += deltaW
            E = np.multiply(gamma * _lambda, E)

            if total_steps % 500 == 0:
                policy_evaluate = policy_eval(W, env, with_discount=EVAL_WITH_DISCOUNT)
                policy_vals.append((total_steps, policy_evaluate))
                print(f'done {total_steps} steps, current policy value is {policy_evaluate}')

                if is_saving_weights and policy_evaluate > best_policy_val:
                    with open(BEST_WEIGHTS_PATH, 'wb') as fd:
                        fd.write(pickle.dumps(W))

            action = new_action
            features = new_features.copy()

            if done:
                break
        
        if is_decay:
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-0.005 * k)

        if total_steps > total_max_steps:
            break

    return W, policy_vals


def get_next_best_action(state, W):
    features = get_features(state)
    return get_Q(features, W).argmax()


def policy_eval(W, env, with_discount=False):
    """
    policy should be an iterable with length of number of states (action per state)
    """
    rewards = []
    for i in range(NUMBER_OF_EVAL_SIMS):
        state = normalize_state(env.reset())
       
        run_reward = 0
        is_done = False
        steps = 0
        while not is_done:
            state, reward, is_done, _ = env.step(get_next_best_action(state, W))
            state = normalize_state(state)
            steps += 1
            if with_discount:
                run_reward += reward * (gamma ** steps)
            else:
                run_reward += reward

        rewards.append(run_reward)

    return np.mean(rewards)


def show_sim_in_env(env, W):
    state = normalize_state(env.reset())
    env.render()
    total_reward = 0
    is_done = False
    num_of_steps = 0

    while not is_done:
        action = get_next_best_action(state, W)
        state, step_reward, is_done, _ = env.step(action)
        state = normalize_state(state)
        total_reward += step_reward

        num_of_steps += 1
        env.render()


def run_and_create_plot(env):
    env.reset()

    W, policy_vals = sarsa_lambda(env, episodes, max_steps_in_episode, alpha=alpha, _lambda=_lambda)
    # show_sim_in_env(env, W)

    plt.figure(figsize=(12, 7))
    plt.plot([x for x, _ in policy_vals], [y for _, y in policy_vals])

    plt.xlabel('number of steps')
    plt.ylabel('mean reward')
    plt.title('Reward As A Function of Number of Steps')
    plt.show()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500

    with open(BEST_WEIGHTS_PATH, 'rb') as fd:
        prev_good_W = pickle.loads(fd.read())

    show_sim_in_env(env, prev_good_W)

    print('calculating policy... (may take a few minutes)')
    run_and_create_plot(env)