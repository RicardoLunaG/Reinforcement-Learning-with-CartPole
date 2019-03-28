import math
import random
import gym
import numpy as np
import tensorflow as tf

class A2C:
    def __init__(self, param_critic, param_actor, action_size, space_size):
        self.time_step = 0
        self.action_size = action_size
        self.space_size = space_size

        self.Actor(**param_actor)
        self.Critic(**param_critic)


        self.init_session()
    #Critic Network, calculates values
    def Critic(self, gamma, hidden_nodes, learning_rate):

        self.input_critic = tf.placeholder(tf.float32,[1,self.space_size],name ="Inputs_Critic")
        self.target_critic = tf.placeholder(tf.float32,[1,1],name = "Target_Critic")
        self.gamma = gamma
        self.l1_critic = tf.layers.dense(
            inputs = self.input_critic,
            units = hidden_nodes,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer()
        )
        self.value = tf.layers.dense(
            inputs = self.l1_critic,
            units = 1,
            activation = None,
            kernel_initializer=tf.truncated_normal_initializer()
        )

        self.loss_critic = tf.reduce_sum(tf.square(self.target_critic-self.value))
        
        self.critic_train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_critic)
    
    #Actor Network, caculates actions probabilities
    def Actor(self, hidden_nodes,learning_rate):

        self.input_actor = tf.placeholder(tf.float32, [1,self.space_size], name = "Input_Actor")
        self.adventage = tf.placeholder(tf.float32, [1,self.action_size], name = "Actions_Actor") #Advantage of taking action

        self.l1_actor = tf.layers.dense(
            inputs = self.input_actor,
            units = hidden_nodes,
            activation= tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer() 
        )
        self.l2_actor = tf.layers.dense(
            inputs = self.l1_actor,
            units = self.action_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer()
        )
        
        self.policy = tf.nn.softmax(self.l2_actor)
        
        self.loss_actor = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.l2_actor,labels=self.adventage))
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_actor)
        tf.summary.scalar("Loss_Actor",self.loss_actor)
    
    def choose_action(self,state):

        pi = self.session.run(self.policy, feed_dict={self.input_actor:state}).ravel()
        return np.random.choice(range(pi.shape[0]),1,p=pi)[0]
    
    def train(self, state, action, reward, next_state, done):

        target_critic = np.zeros([1,1])
        target_actor = np.zeros([1,self.action_size])

        V = self.session.run(self.value, feed_dict={self.input_critic:state})
        V_next = self.session.run(self.value, feed_dict={self.input_critic:next_state})

        if done:
            target_critic[0][0] = reward
            target_actor[0][action] = reward - V
        else:
            target_critic[0][0] = reward + self.gamma * V_next
            target_actor[0][action] = reward + self.gamma * V_next - V
        
        _,_ = self.session.run(
            [self.critic_train_op, self.actor_train_op],
            feed_dict={
                self.input_actor:state,
                self.input_critic:state,
                self.target_critic:target_critic,
                self.adventage:target_actor
            }
        )

        self.time_step+=1
        
   
    def init_session(self):
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

NUMBER_EPISODES = 3000
NUMBER_CONSECUTIVE_EPISODES = 20
REWARD_TO_REACH = 490
solved = False
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
penalize_bad_states = (True,-100)

#Change to True if render is wanted
render = False

#Parameters for the Critic Network
param_critic = {
    'gamma':1,
    'hidden_nodes': 200,
    'learning_rate': 0.001
}

#Parameters for the Actor Network
param_actor = {
    'hidden_nodes': 200,
    'learning_rate': 0.001
}

def main():
    a2c = A2C(param_critic,param_actor,action_size,state_size)

    for i in range(NUMBER_EPISODES):
        state = env.reset().reshape([1,4])

        done = False
        t = 0
        while not done:
            t+=1

            if render:
                env.render()
            
            action = a2c.choose_action(state)
            next_state,reward,done,_ = env.step(action)

            next_state = next_state.reshape([1,4])
            if penalize_bad_states[0] and done and t < 500:
                reward = penalize_bad_states[1]

            a2c.train(state,action,reward,next_state,done and t < 500)
            state = next_state
        
        
        print('Episode {}/{}  Reward: {}'.format(i,NUMBER_EPISODES,t))

if __name__ == '__main__':
    main()