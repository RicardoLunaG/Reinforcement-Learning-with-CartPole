import gym
import numpy as np
import tensorflow as tf
import copy

NUMBER_EPISODES = 10000
GAMMA = 0.95

class ACModel:
    def __init__(self,scope:str,env,temp = 0.1):

        ob_space = env.observation_space.shape[0]
        action_size = env.action_space.n

        with tf.variable_scope(scope):
            self.states = tf.placeholder(dtype=tf.float32,shape=[None,ob_space],name = "States")
            
            with tf.variable_scope('Policy_Estimator'):
                l1 = tf.layers.dense(inputs=self.states,units = 200, activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
                l2 = tf.layers.dense(inputs=l1,units = 200, activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.action_probs = tf.layers.dense(inputs=tf.divide(l2,temp), units=action_size, activation=tf.nn.softmax)
                
            with tf.variable_scope('Value_Estimator'):
     
                l1 = tf.layers.dense(inputs=self.states, units = 200, activation=tf.nn.relu,kernel_initializer=tf.variance_scaling_initializer(scale=2))
                l2 = tf.layers.dense(inputs=l1, units = 200, activation=tf.nn.relu,kernel_initializer=tf.variance_scaling_initializer(scale=2))
                self.value = tf.layers.dense(inputs = l2, units = 1, activation = None)
           
            
            self.act_stochastic = tf.multinomial(tf.log(self.action_probs), num_samples = 1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.action_probs, axis = 1)
            
            self.scope = tf.get_variable_scope().name

    def act(self, states, stochastic = True):
        sess = tf.get_default_session()
        if stochastic:
            return sess.run([self.act_stochastic,self.value], feed_dict={self.states:states})
        else:
            return sess.run([self.act_deterministic,self.value], feed_dict={self.states:states})

    def get_action_probs(self,states):
        sess = tf.get_default_session()
        return sess.run(self.action_probs, feed_dict = {self.states:states})
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class PPOTrain:
    def __init__(self,Policy, Old_Policy, gamma = 0.95, clip_value = 0.2, c_1 = 1, c_2=0.01):
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()
        
        "Updater"
        self.assign_ops = []
        for v_old, v in zip(old_pi_trainable, pi_trainable):
            self.assign_ops.append(tf.assign(v_old,v))
        

        self.actions = tf.placeholder(tf.int32, [None], "Actions")
        self.rewards = tf.placeholder(tf.float32, [None], "Rewards")
        self.v_next = tf.placeholder(tf.float32, [None], "Value_Next")
        self.gaes = tf.placeholder(tf.float32, [None], "GAES")
        
        action_probs = self.Policy.action_probs
        action_probs_old = self.Old_Policy.action_probs

        #Probabilities of actions took with policy
        action_probs = action_probs * tf.one_hot(indices = self.actions, depth = action_probs.shape[1])
        action_probs = tf.reduce_sum(action_probs, axis = 1)

        #Probabilites of action took with old policy
        action_probs_old = action_probs_old * tf.one_hot(indices = self.actions, depth = action_probs_old.shape[1])
        action_probs_old = tf.reduce_sum(action_probs_old, axis = 1)

        "Clipped Policy Loss"
        ratios = tf.exp(tf.log(action_probs)-tf.log(action_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min = 1 - clip_value, clip_value_max = 1 + clip_value)
        loss_clip = tf.minimum(tf.multiply(self.gaes,ratios),tf.multiply(self.gaes,clipped_ratios))
        loss_clip = tf.reduce_mean(loss_clip)


        "Value Loss"
        v_preds = self.Policy.value
        loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_next, v_preds)
        loss_vf = tf.reduce_mean(loss_vf)
              
        "Entropy"
        entropy = - tf.reduce_sum(self.Policy.action_probs * tf.log(tf.clip_by_value(self.Policy.action_probs, 1e-10, 1.0)), axis = 1)
        entropy = tf.reduce_mean(entropy, axis = 0)

        "Total Loss"
        loss = loss_clip - c_1 * loss_vf + c_2 * entropy
        loss = -loss
        
        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon = 1e-5)
        self.train_op = optimizer.minimize(loss, var_list = pi_trainable)
    
    def train(self, states, actions, rewards, v_next, gaes):
        sess = tf.get_default_session()
        sess.run([self.train_op], feed_dict={
            self.Policy.states:states,
            self.Old_Policy.states:states,
            self.actions: actions,
            self.rewards: rewards,
            self.v_next: v_next,
            self.gaes: gaes
        })
    def get_summary(self, states, actions, rewards, v_next, gaes):
        sess = tf.get_default_session()
        return sess.run([self.merged], feed_dict = {
            self.Policy.states: states,
            self.Old_Policy.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.v_next: v_next,
            self.gaes: gaes
        })
    def assign_policy_parameters(self):
        sess = tf.get_default_session()
        sess.run(self.assign_ops)
    #Generative Advantage Estimador
    def get_gaes(self,rewards, values, v_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_next, values)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes)-1)):
            gaes[t] = gaes[t] + self.gamma * gaes[t+1]
        return gaes

def main():
    env = gym.make('CartPole-v1')
    env.seed(0)
    state_space = env.observation_space
    Policy = ACModel('Policy', env)
    Old_Policy = ACModel('Old_Policy', env)
    PPO = PPOTrain(Policy,Old_Policy,GAMMA)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        reward = 0

        for iteration in range(NUMBER_EPISODES):
            states = []
            actions = []
            values = []
            rewards = []
            t = 0
            state = env.reset()
            done = False
            while not done:
                t += 1
                state = np.stack([state]).astype(dtype=np.float32)
                action, value = Policy.act(state, True)

                action = np.asscalar(action)
                value = np.asscalar(value)
                
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                values.append(value)
                
                if done: 
                    v_next = values[1:] + [0]
                    reward = -10

                
                rewards.append(reward)

                state = next_state

            gaes = PPO.get_gaes(rewards, values, v_next)

            states = np.reshape(states, newshape = [-1]+list(state_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_next = np.array(v_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            PPO.assign_policy_parameters()
            inp = [states, actions, rewards, v_next, gaes]
            
            #For sample efficiency 
            for _ in range(4):
                sample_indices = np.random.randint(low  = 0, high = states.shape[0], size = 64)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis = 0) for a in inp]
                PPO.train(sampled_inp[0],sampled_inp[1],sampled_inp[2], sampled_inp[3], sampled_inp[4])

            print("Episode: {}/{} Reward:{}".format(iteration,NUMBER_EPISODES,sum(rewards)))

if __name__ == '__main__':
    main()