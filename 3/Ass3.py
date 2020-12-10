import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from itertools import product

alpha = 0.02
max_steps = 500
episodes = 1000000
gamma = 0.95
epsilon_max = 1.0
epsilon_min = 0.1
_lambda = 0.5
NUMBER_OF_EVAL_SIMS = 100
EVAL_WITH_DISCOUNT = False

ACTION_NO = 3
# -1.2 is the leftest position.
# The env starts at a location between -0.6 and -0.4 randomly.
# The agents win when he gets to 0.5
C_P = [-1.2, -0.6, -0.4, 0.5]
# -0.7 is the min speed, 0.7 is the max speed.
C_VEL = list(np.linspace(-0.7, 0.7, 8))
N = len(C_P) * len(C_VEL)

#ToDo:maybe change this
P = len(C_P)
V = len(C_VEL)
shape = (P,V,ACTION_NO)

def eps_greedy_policy(epsilon,state,Q, env):
    if random.uniform(0, 1) < epsilon:
        new_action = env.action_space.sample()  # explore
    else:
        p,v = state
        new_action = np.argmax(Q[p,v, :])  # exploit
    return new_action

def init_Weights():   
    return np.zeros((N,ACTION_NO))


def init_E(W):
    return np.zeros((P,V))


def init_Q():
    return np.zeros(shape)


def get_features(p, v):
    """
    returns a vector phi that each entry is computed feature for given p, v
    """
    # reshape to make it a matrix with one row (so we can transpose it later)
    prod = product(C_P, C_VEL)
    C = [np.array(val).reshape((1, -1)) for val in prod]
    p_v = np.array([p, v]).reshape((1, -1)).T
    X = np.array([p_v - c_entry.T for c_entry in C])
    inv_cov = np.linalg.inv(np.diag([0.04, 0.0004]))
    phi = np.array([np.e ** (-(xi.T @ inv_cov @ xi) / 2) for xi in X])
    return np.squeeze(phi)   # get rid of 2 unnecessary dimensions


def stochasticGradient(p,v,W):
    return get_features(p,v)


# float p, float v -> p_index,v_index
def map_p_v(observation,env):
    low = env.observation_space.low #2 dimension
    high = env.observation_space.high #2 dimension
    size = (high-low) 

    p = int(((observation[0]-low[0])/size[0])*P)
    v = int(((observation[1]-low[1])/size[1])*V)

    return p,v

def get_Q(p,v,action,W_a):
    return get_features(p, v) @ W_a


def sarsa_lambda(env, episodes=episodes, max_steps=max_steps,
                 epsilon_max=epsilon_max, epsilon_min=epsilon_min, is_decay=True, _lambda=_lambda, alpha=alpha,
                 q_approx_func=None):

    # (p,v,a) = (p,v,i) @ (i,a) => the shape of W should be (i,a): (32,3) 
    W = init_Weights()
    
    total_steps = 0
    epsilon = epsilon_max
    policy_vals = []

    for k in range(episodes):
        # init E,S,A
        E = init_E(W)
        #state = (pos,vel)
        state = env.reset()
        Q =  init_Q()
        action = eps_greedy_policy(epsilon, map_p_v(state,env), Q, env)

        for step in range(max_steps):
            # Take action A, obvserve R,S'
            new_state, reward, done, _ = env.step(action)
            #state = (pos,vel)
            new_action = eps_greedy_policy(epsilon, map_p_v(new_state,env), Q, env)
            p,v = new_state
            delta_error = reward + gamma * get_Q(p,v,new_action,W[:,new_action]) - get_Q(p,v,action,W[:,action])
            E[map_p_v(new_state,env)] += 1 # E suppose to be like W shape, so i am not sure what should represent the first dimension
            E = np.multiply(gamma * _lambda, E) + (stochasticGradient(p,v,W)).reshape((P,V))
            deltaW = (np.multiply(alpha*delta_error,E)).reshape(P*V)
    
            W[:,action]+=deltaW    
            state = new_state
            action = new_action
            total_steps += 1

            #ToDo: play with number of steps
            if (total_steps < 20000 and total_steps % 2000 == 0) or (total_steps >= 20000 and total_steps % 7000 == 0):
                policy = np.argmax(Q, axis=2)
                policy_evaluate = policy_eval(policy, env, with_discount=EVAL_WITH_DISCOUNT)
                policy_vals.append((total_steps, policy_evaluate))

            if done:
                break
        
        if is_decay:
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-0.00005 * k)

        if total_steps > 1e6:
            break

    for p in range(N):
        for v in range(N):
            for a in range(ACTION_NO):
                Q[p,v,a] = get_Q(p,v,action,W[a])

    return Q, np.argmax(Q, axis=2), policy_vals  # returns Q, the policy and the values of the policy during the run


def policy_eval(policy, env, with_discount=False):
    """
    policy should be an iterable with length of number of states (action per state)
    """
    rewards = []
    for i in range(NUMBER_OF_EVAL_SIMS):
        state = env.reset()
        state = map_p_v(state,env)

        run_reward = 0
        is_done = False
        steps = 0
        while not is_done:
            state, reward, is_done, _ = env.step(policy[state])
            state = map_p_v(state,env)
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
    env = gym.make('MountainCar-v0')
    run_and_create_plot(env)