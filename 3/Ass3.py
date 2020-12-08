import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from itertools import product


alpha = 0.05
max_steps = 500
episodes = 1000000
gamma = 0.95
epsilon_max = 1.0
epsilon_min = 0.1
_lambda = 0.9
NUMBER_OF_EVAL_SIMS = 100
EVAL_WITH_DISCOUNT = False

ACTION_NO = 3
#Number of Features
#ToDo: figure out what suppose to be total number of features
N = 50 
shape = (N,N,ACTION_NO,N)

def eps_greedy_policy(epsilon,state,Q, env):
    if random.uniform(0, 1) < epsilon:
        new_action = env.action_space.sample()  # explore
    else:
        p,v = state
        new_action = np.argmax(Q[p,v, :])  # exploit
    return new_action


#ToDo: maybe initilize the weight with different values and maybe change the shape
def initWeights():
    #shape -> pos,vel,action, number of features
    return np.zeros(shape)


def init_E():
    return np.zeros(shape)


def init_Q():
    return np.zeros(shape)


#ToDo: implement methods
def initFeatures():
    #shape -> pos,vel,action, number of features
    # return np.zeros(shape)
    raise NotImplementedError

def stochasticGradient(Q_w):
    raise NotImplementedError

# float p, float v -> p_index,v_index
def map_p_v(observation,env):
    low = env.observation_space.low #2 dimension
    high = env.observation_space.high #2 dimension
    dx = (high-low) / N

    p = int((observation[0]-low[0])/dx[0])
    v = int((observation[1]-low[1])/dx[1])

    return p,v

def get_Q_w(X,state,action,W):
    p,v = state
    return X[p,v,action] @ W[p,v,action]


def sarsa_lambda(env, episodes=episodes, max_steps=max_steps,
                 epsilon_max=epsilon_max, epsilon_min=epsilon_min, is_decay=True, _lambda=_lambda, alpha=alpha,
                 q_approx_func=None):

    X = initFeatures()
    W = initWeights() # W_j -> (pos,vel,action)
    total_steps = 0
    epsilon = epsilon_max
    policy_vals = []

    for k in range(episodes):
        
        # init E,S,A
        E = init_E()
        #state = (pos,vel)
        observation = env.reset()
        state = map_p_v(observation,env)
        Q_w =  init_Q()
        action = eps_greedy_policy(epsilon, state, Q_w, env)

        for step in range(max_steps):
            # Take action A, obvserve R,S'
            observation, reward, done, _ = env.step(action)
            new_state = map_p_v(observation,env)
            #state = (pos,vel)
            new_action = eps_greedy_policy(epsilon, new_state, Q_w, env)
            Q_w_new = get_Q_w(X,new_state,new_action,W)

            delta_error = reward + gamma * Q_w_new - Q_w
            E[state, action] += 1
            Q_w = np.add(Q_w, np.multiply(alpha * delta_error, E))
            E = np.multiply(gamma * _lambda, E) + stochasticGradient(Q_w)
            deltaW = np.multiply(alpha*delta_error,E)
        
            W+=deltaW    
            state = new_state
            action = new_action
            total_steps += 1

            #ToDo: play with number of steps
            if (total_steps < 20000 and total_steps % 2000 == 0) or (total_steps >= 20000 and total_steps % 7000 == 0):
                policy = np.argmax(Q_w, axis=2)
                policy_evaluate = policy_eval(policy, env, with_discount=EVAL_WITH_DISCOUNT)
                policy_vals.append((total_steps, policy_evaluate))

            if done:
                break
        
        if is_decay:
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-0.00005 * k)

        if total_steps > 1e6:
            break

    return Q_w, np.argmax(Q_w, axis=2), policy_vals  # returns Q, the policy and the values of the policy during the run


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
    env = gym.make('MountainCar-v0')
    run_and_create_plot(env)