import numpy as np
import gym
import math
from time import sleep

env = gym.make('CartPole-v1')

NUMBER_EPISODES = 2000
step_size = 0.5
min_step_size = 0.1
min_e = 0.1
discount = 1
limiters = (1,1,6,12)

Q = np.zeros(limiters+(env.action_space.n,)) 

#State space discreatization is neccesary for continous state spaces when tables are being used.
def featureDis(obs):
    up_bounds = [env.observation_space.high[0], 0.5,env.observation_space.high[2],math.radians(50)]
    low_bounds = [env.observation_space.low[0], -0.5,env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(low_bounds[i]))/(up_bounds[i]-low_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((limiters[i]-1)*ratios[i])) for i in range(len(obs))]
    new_obs = [min(limiters[i]-1, max(0,new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def greedy_action(a,i):
    r = np.random.random()
    
    e = max(min_e, min(1, 1.0 - math.log10((i + 1) / 25)))
    if r < e:
        return env.action_space.sample()
    else:
        return a

render = False

def main():
    for i in range(NUMBER_EPISODES):
        state = featureDis(env.reset())
        action = greedy_action(np.argmax(Q[state]),i)
        done = False
        step_size = max(min_step_size, min(0.5, 1.0 - math.log10((i + 1) / 25)))
        t = 0
        while not done:
            t += 1
            if render:
                env.render()

            new_state, reward, done, _ = env.step(action)
            new_state = featureDis(new_state)
            new_action = greedy_action(np.argmax(Q[new_state]),i)
            future_Q = Q[new_state][new_action]
            
            Q[state][action] = Q[state][action] + step_size*(reward+discount*future_Q-Q[state][action])
            state = new_state
            action = new_action

        print('Episode {}/{}  Reward: {}'.format(i,NUMBER_EPISODES,t))




if __name__ == '__main__':
    main()




