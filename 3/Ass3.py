import numpy as np
import gym
import random
#import matplotlib.pyplot as plt
from itertools import product

DEBUG = True
np.set_printoptions(precision=2)


alpha = 0.02
max_steps = 500
episodes = 1000000
gamma = 0.95
epsilon_max = 0.1
epsilon_min = 0.005
_lambda = 0.5
NUMBER_OF_EVAL_SIMS = 100
EVAL_WITH_DISCOUNT = False
ACTION_NO = 3


# -1.2 is the leftest position.
# The env starts at a location between -0.6 and -0.4 randomly.
# The agents win when he gets to 0.5
# C_P = [-1.2, -0.6, -0.4, 0.5]
C_P = list(np.linspace(-1.2, 0.6, 4))
# -0.7 is the min speed, 0.7 is the max speed.
C_VEL = list(np.linspace(-0.7, 0.7, 8))
N = len(C_P) * len(C_VEL)

prod = product(C_P, C_VEL)
C = [np.array(val).reshape((1, -1)) for val in prod]

C2 = np.array([[0.  , 0.  ],
       [0.  , 0.33],
       [0.  , 0.67],
       [0.  , 1.  ],
       [0.33, 0.  ],
       [0.33, 0.33],
       [0.33, 0.67],
       [0.33, 1.  ],
       [0.67, 0.  ],
       [0.67, 0.33],
       [0.67, 0.67],
       [0.67, 1.  ],
       [1.  , 0.  ],
       [1.  , 0.33],
       [1.  , 0.67],
       [1.  , 1.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ],
       [0.  , 0.  ]])
N2 = 64

def init_Weights():   
    return np.zeros((N2,ACTION_NO))


def init_E():
    return np.zeros((N2,ACTION_NO))


def get_features(_state):
    _phi = np.zeros(N2)
    for _k in range(N2):
        _phi[_k] = np.exp(-np.linalg.norm(_state - C2[_k, :]) ** 2 / 0.05555555555555555)
    return _phi


def get_features2(state):
    """
    returns a vector phi that each entry is computed feature for given p, v
    """
    # reshape to make it a matrix with one row (so we can transpose it later)
    p,v=state
    p_v = np.array([p, v]).reshape((1, -1)).T
    X = np.array([p_v - c_entry.T for c_entry in C])
    inv_cov = np.linalg.inv(np.diag([0.04, 0.0004]))
    # phi = np.array([np.e ** (-(xi.T @ inv_cov @ xi) / 2) for xi in X])
    phi = np.array([np.exp (-(xi.T @ inv_cov @ xi) / 2) for xi in X])

    return np.squeeze(phi)   # get rid of 2 unnecessary dimensions

def eps_greedy_policy(epsilon,Q):
    if random.uniform(0, 1) < epsilon:
        new_action = env.action_space.sample()  # explore
    else:
        new_action = Q.argmax()  # exploit
    return new_action


# Returns the state scaled between 0 and 1
def normalize_state(state):
    p,v = state
    p_n = (p-env.observation_space.low[0]) / (env.observation_space.high[0] - env.observation_space.low[0])
    v_n = (v-env.observation_space.low[1]) / (env.observation_space.high[1] - env.observation_space.low[1])

    return p_n,v_n

#p,v should be normalized
def get_Q(features,W):
    return np.array([features @ W[:,a] for a in range(ACTION_NO)])


def get_Q_a(features,W_a):
    return features @ W_a


def sarsa_lambda(env, episodes=episodes, max_steps=max_steps,
                 epsilon_max=epsilon_max, epsilon_min=epsilon_min, is_decay=False, _lambda=_lambda, alpha=alpha):

    # (p,v,a) = (p,v,i) @ (i,a) => the shape of W should be (i,a): (32,3) 
    W = init_Weights()
    total_steps = 0
    epsilon = epsilon_max

    if DEBUG:
        first_pass = False
    
    for k in range(episodes):
        # init E,S,A
        #state = (pos,vel)
        E = init_E()
        state = normalize_state(env.reset()) 
        features = get_features(state)
        action = eps_greedy_policy(epsilon,get_Q(features,W))

        for step in range(max_steps):
            total_steps += 1
            if DEBUG:
                if first_pass:
                    env.render()
            # Take action A, obvserve R,S'
            new_state, reward, done, _ = env.step(action)
            new_state = normalize_state(new_state)
            new_features = get_features(new_state)
            new_action = eps_greedy_policy(epsilon,get_Q(new_features,W))
            curr_Q_p_v_a = get_Q_a(features,W[:,action])
            next_Q_p_v_a = get_Q_a(new_features,W[:,new_action])
            
            if done:
                delta_error = reward  - curr_Q_p_v_a
                if DEBUG:
                     if total_steps % 500 != 0:
                         first_pass = True
                     print("total steps = ",total_steps)
            else:
                delta_error = reward + gamma * next_Q_p_v_a  - curr_Q_p_v_a

            stochasticGradient = features
            # E[:, action] +=  stochasticGradient  # accumulating traces
            E[:, action] = stochasticGradient  # replacing traces
           
            deltaW = (np.multiply(alpha*delta_error,E))
            W+=deltaW
            E = np.multiply(gamma * _lambda, E)
            

            #ToDo: play with number of steps
            # if (total_steps < 20000 and total_steps % 2000 == 0) or (total_steps >= 20000 and total_steps % 8000 == 0):
            #     policy_evaluate = policy_eval(W, env, with_discount=EVAL_WITH_DISCOUNT)                
            #     policy_vals.append((total_steps, policy_evaluate))
                

            state = new_state
            action = new_action
            features = new_features.copy()

            if done:
                break
        
        if is_decay:
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-0.005 * k)

        if total_steps > 1e6:
            break

    return W


def get_policy(state,W):
    features = get_features(state)
    return get_Q(features,W).argmax()


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
            state, reward, is_done, _ = env.step(get_policy(state,W))
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
        action = np.argmax(get_policy(state,W))
        state, step_reward, is_done, _ = env.step(action)
        state = normalize_state(state)
        total_reward += step_reward

        num_of_steps += 1
        env.render()

    if DEBUG:
        print('done in {} steps and reward: {}'.format(num_of_steps, total_reward))


def run_and_create_plot(env):
    global alpha,_lambda
    alphas = [alpha]
    lambdas = [_lambda]
    #plt.figure(figsize=(12, 7))

    for alpha, _lambda in product(alphas, lambdas):
        description = 'alpha: {}, lambda: {}'.format(alpha, _lambda)
        print(description)

        W = sarsa_lambda(env, episodes, max_steps, alpha=alpha, _lambda=_lambda)
        show_sim_in_env(env, W)

        # plt.plot([x for x, _ in policy_vals], [y for _, y in policy_vals],
        #          label=description, alpha=0.6)

    # plt.xlabel('number of steps')
    # plt.ylabel('avarage reward')
    # plt.title('Reward For Different Hyper Parameters')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    run_and_create_plot(env)