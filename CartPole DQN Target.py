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
NUM_STEPS_TO_UPDATE = 50

class DoubleNetwork:
    def __init__(self, main,target,sess):
        self.main = main
        self.target = target
        self.memory = Memory(10000)
        self.eps = MAX_EPSILON
        self.steps = 0
        self.sess = sess

    def choose_action(self, state):
        if random.random() < self.eps:
            action = random.randint(0, self.main.num_actions - 1)
        else:
            action = np.argmax(self.target.predict_one(state,self.sess))
        self.steps += 1
        self.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA*self.steps)
        return action
    def replay(self):
        batch = self.memory.sample(self.main.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self.main.num_states) if val[3] is None else val[3]) for val in batch])
        q_s_a = self.main.predict_batch(states, self.sess)
        q_s_a_d = self.target.predict_batch(next_states, self.sess)
        x = np.zeros((len(batch), self.main.num_states))
        y = np.zeros((len(batch), self.main.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            current_q = q_s_a[i]
            if next_state is None:
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self.main.train_batch(self.sess, x, y)
    def update_target(self):
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_MAIN")
        q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_TARGET")
        assert len(q_vars) == len(q_target_vars)
        self.sess.run([v_t.assign(v) for v_t, v in zip(q_target_vars, q_vars)])

class Model:
    def __init__(self, num_states,num_actions,batch_size,scope):
        with tf.variable_scope(scope):
            self.num_states = num_states
            self.num_actions = num_actions
            self.batch_size = batch_size
            self.eps = MAX_EPSILON
            self.steps = 0

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
    main = Model(num_states, num_actions, BATCH_SIZE, "Q_MAIN")
    target = Model(num_states, num_actions, BATCH_SIZE, "Q_TARGET")
    

    with tf.Session() as sess:
        sess.run(main.var_init)
        sess.run(target.var_init)
        dobuleNet = DoubleNetwork(main, target, sess)
        t = 0
        for i in range(NUM_EPISODES):

            state = env.reset()
            total_rewad = 0
            done = False
            while not done:
                t += 1
                if render:
                    env.render()
                
                action = dobuleNet.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                total_rewad += reward
                
                if done:
                    next_state = None

                dobuleNet.memory.add_sample((state,action,reward,next_state))
                dobuleNet.replay()

                state = next_state
                if t % NUM_STEPS_TO_UPDATE == 0:
                    dobuleNet.update_target()
            print("Episode: {}/{} Reward:{}".format(i,NUM_EPISODES,total_rewad))


if __name__ == "__main__":
    main()

    
