import gym
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50
NUM_EPISODES = 2000

class Model:
    def __init__(self, num_states,num_actions,batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.eps = MAX_EPSILON
        self.steps = 0
        self.memory = Memory(10000)
        self.states = tf.placeholder(shape=[None,self.num_states],dtype=tf.float32)
        self.q_s_a = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        
        fc1 = tf.layers.dense(self.states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1,50,activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2,self.num_actions)
        loss = tf.losses.mean_squared_error(self.q_s_a,self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self.states: state.reshape(1, self.num_states)})
    
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self.states: states})

    def train_batch(self, sess, x_batch, y_batch):
        return sess.run(self._optimizer, feed_dict={self.states:x_batch,self.q_s_a:y_batch})

    def choose_action(self, state, sess ):
        if random.random() < self.eps:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.predict_one(state,sess))
        self.steps += 1
        self.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA*self.steps)

        return action
    
    def replay(self, sess):
        batch = self.memory.sample(self.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self.num_states) if val[3] is None else val[3]) for val in batch])

        q_s_a = self.predict_batch(states, sess)

        q_s_a_d = self.predict_batch(next_states, sess)

        x = np.zeros((len(batch), self.num_states))
        y = np.zeros((len(batch), self.num_actions)) 
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]

            current_q = q_s_a[i]

            if next_state is None:
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self.train_batch(sess, x, y)
    

class Memory:
    def __init__(self, max_memmory):
        self._max_memory = max_memmory
        self._samples = []

    def add_sample(self,sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, num_samples):
        if num_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, num_samples)


 
def main():
    env = gym.make("CartPole-v1")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    render = False
    model = Model(num_states, num_actions, BATCH_SIZE)

    with tf.Session() as sess:
        sess.run(model.var_init)
        
        for i in range(NUM_EPISODES):

            state = env.reset()
            total_rewad = 0
            done = False
            while not done:
                if render:
                    env.render()
                
                action = model.choose_action(state,sess)
                next_state, reward, done, _ = env.step(action)

                total_rewad += reward
                
                if done:
                    next_state = None

                model.memory.add_sample((state,action,reward,next_state))
                model.replay(sess)

                state = next_state
            print("Episode: {}/{} Reward:{}".format(i,NUM_EPISODES,total_rewad))


if __name__ == "__main__":
    main()

    
