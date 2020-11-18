import numpy as np
import gym

env = gym.make('Taxi-v3')
check_states = []
gamma=0.95
delta=0.001

def eval_reward_prob(env):
    R = np.zeros((env.env.nS,env.env.nA,env.env.nS)) #state,action,next state
    T = np.zeros((env.env.nS,env.env.nA,env.env.nS)) #state,action,next state
    for state in range(env.env.nS):
        for action in range(env.env.nA):
            for prob, next_state, reward, _ in env.env.P[state][action]:
                R[state, action, next_state] = reward
                T[state, action, next_state] = prob
    
    return R,T

def policy_iteration(env,gamma=0.95,delta=0.001,max_iterations=100000):
    #init V with zeros,policy with random values,R and P from env
    policy = np.array([env.action_space.sample() for _ in range(env.env.nS)])
    V = np.zeros(env.env.nS)
    R,T = eval_reward_prob(env)

    for _ in range(max_iterations):
        prev_V = V.copy()
        Q = (T * (R + gamma * V)).sum(axis=2) #action-value their next state using bellman equation
        bin_policy = np.zeros((env.env.nS,env.env.nA))
        bin_policy[np.arange(env.env.nS),policy] = 1
        V = np.sum(bin_policy*Q,axis = 1) #sum up all actions by given policy,s to receive value function

        if np.max(np.abs(prev_V - V)) < delta:
            break
        policy = np.argmax(Q,axis=1)
        
    return policy,V


policy,V = policy_iteration(env,gamma,delta)

def simulateGame(MAX_STEPS = 100):
    total_reward = 0
    state = env.reset()

    for steps in range(MAX_STEPS):
        env.render()
        action = policy[state]
        state,reward,done,_ = env.step(action)
        total_reward+=reward
        print("Reward: ",reward)

        if done:
            print("Done!!!")
            print("Score:{} Steps:{}".format(total_reward,steps))
            break

    return total_reward

def valueFunction(s): 
    return V[s]
    
simulateGame()
v_s  = np.array([valueFunction(state) for state in range (env.env.nS)])
print(v_s)







        