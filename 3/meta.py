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

#C_P = list(np.linspace(-1.2, 0.6, 4))
# -0.7 is the min speed, 0.7 is the max speed.
#C_VEL = list(np.linspace(-0.7, 0.7, 8))
#N = len(C_P) * len(C_VEL)

#prod = product(C_P, C_VEL)
#C = [np.array(val).reshape((1, -1)) for val in prod]
# C2 = np.array([[0.  , 0.  ],
#        [0.  , 0.33],
#        [0.  , 0.67],
#        [0.  , 1.  ],
#        [0.33, 0.  ],
#        [0.33, 0.33],
#        [0.33, 0.67],
#        [0.33, 1.  ],
#        [0.67, 0.  ],
#        [0.67, 0.33],
#        [0.67, 0.67],
#        [0.67, 1.  ],
#        [1.  , 0.  ],
#        [1.  , 0.33],
#        [1.  , 0.67],
#        [1.  , 1.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ],
#        [0.  , 0.  ]])

N = 32
# N2 = 64


good_centers = []

def init_Weights():   
    return np.zeros((N,ACTION_NO))


def init_E():
    return np.zeros((N,ACTION_NO))


# def get_features2(_state):
#     _phi = np.zeros(N2)
#     for _k in range(N2):
#         _phi[_k] = np.exp(-np.linalg.norm(_state - C2[_k, :]) ** 2 / 0.05555555555555555)
#     return _phi


def get_features(state):
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

    W = init_Weights()
    total_steps = 0
    epsilon = epsilon_max

    d = 0
    
    for k in range(episodes):
        # init E,S,A
        #state = (pos,vel)
        E = init_E()
        state = normalize_state(env.reset()) 
        features = get_features(state)
        action = eps_greedy_policy(epsilon,get_Q(features,W))

        for step in range(max_steps):
            total_steps += 1
            # Take action A, obvserve R,S'
            new_state, reward, done, _ = env.step(action)
            new_state = normalize_state(new_state)
            new_features = get_features(new_state)
            new_action = eps_greedy_policy(epsilon,get_Q(new_features,W))
            curr_Q_p_v_a = get_Q_a(features,W[:,action])
            next_Q_p_v_a = get_Q_a(new_features,W[:,new_action])
            
            if done:
                delta_error = reward  - curr_Q_p_v_a
                if step + 1 < max_steps:
                    d+=1

            else:
                delta_error = reward + gamma * next_Q_p_v_a  - curr_Q_p_v_a

            stochasticGradient = features
            # E[:, action] +=  stochasticGradient  # accumulating traces
            E[:, action] = stochasticGradient  # replacing traces
           
            deltaW = (np.multiply(alpha*delta_error,E))
            W+=deltaW
            E = np.multiply(gamma * _lambda, E)
            
            state = new_state
            action = new_action
            features = new_features.copy()

            if done:
                break
        
        if is_decay:
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-0.005 * k)

        if total_steps > 30000:
            global good_centers
            good_centers = np.append(good_centers,np.array([C_P,C_VEL,d]))
            if d > 0:
                print("found good random centers. d: ",d)
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


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    global C

    while True:
        C_P = list(np.random.uniform(-1.2,0.6,4))
        C_VEL = list(np.random.uniform(-0.7,0.7,8))
        prod = product(C_P, C_VEL)
        C = [np.array(val).reshape((1, -1)) for val in prod]
        W = sarsa_lambda(env, episodes, max_steps, alpha=alpha, _lambda=_lambda)
        s = good_centers.reshape((-1,3))
        l = 1
        if len(s) > l:
            a = s[(s[:,2]).argsort()][-l:,0]
            b = s[(s[:,2]).argsort()][-l:,1]
            ah = np.vstack(a)
            bh = np.vstack(b)
            print("pos avg",np.array(sorted(np.mean(ah,axis=0))))
            print("vel avg",np.array(sorted(np.mean(bh,axis=0))))

