import tensorflow as tf
import numpy as np
import gym

env = gym.make("CartPole-v1")
env.unwrapped
env.seed(1)

#Parameters
state_size = 4
action_size = env.action_space.n

NUMBER_EPISODES = 10000
LEARNING_RATE = 0.01
GAMMA = 0.95

def discount_and_normalized_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * GAMMA + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean)/(std)

    return discounted_episode_rewards

class Network:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, state_size],name="Inputs")
        self.actions = tf.placeholder(tf.int32, [None,action_size],name="Actions")
        self.discounted_episode_rewards = tf.placeholder(tf.float32, [None],name="Discounted_Rewards")

        self.mean_reward_ = tf.placeholder(tf.float32, name = "Mean_Reward")

        fc1 = tf.contrib.layers.fully_connected(inputs= self.input,num_outputs = 10, activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,num_outputs=action_size, activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())
        fc3 = tf.contrib.layers.fully_connected(inputs=fc2,num_outputs=action_size,activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer())

        self.action_distribution = tf.nn.softmax(fc3)


        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.actions)
        self.loss = tf.reduce_mean(neg_log_prob*self.discounted_episode_rewards)

        self.train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def get_actions(self,state):
        sess = tf.get_default_session()
        return sess.run(self.action_distribution,feed_dict={self.input:state.reshape([1,4])})
    def train(self, episode_states, episode_actions,discounted_episode_rewards):
        sess = tf.get_default_session()
        sess.run([self.loss,self.train_opt], feed_dict={self.input: np.vstack(np.array(episode_states)),self.actions:np.vstack(np.array(episode_actions)), self.discounted_episode_rewards:discounted_episode_rewards})

def main():


    rewards = []
    episode_states, episode_actions, episode_rewards = [],[],[]

    with tf.Session() as sess:
        
        current = 0
        mean_reward_last = 0
        last_rewards = np.zeros(20)
        network = Network()
        sess.run(tf.global_variables_initializer())

        for i in range(NUMBER_EPISODES):
            rewards_sum = 0
            state = env.reset()
            done = False
            while not done:
                action_probabily_distribution = network.get_actions(state)
                action = np.random.choice(range(action_probabily_distribution.shape[1]),p=action_probabily_distribution.ravel())

                new_state, reward, done, _ = env.step(action)

                episode_states.append(state)

                action_ = np.zeros(action_size)
                action_[action] = 1

                episode_actions.append(action_)

                episode_rewards.append(reward)

                state = new_state

            rewards_sum = np.sum(episode_rewards)
            rewards.append(rewards_sum)
            
            current += 1
            last_rewards[current-1] = rewards_sum
            mean_reward_last = np.mean(last_rewards)
            current = current % 20

            print('Episode {}/{}  Reward: {}'.format(i,NUMBER_EPISODES,rewards_sum))

            discounted_episode_rewards = discount_and_normalized_rewards(episode_rewards)
            
            network.train(episode_states,episode_actions,discounted_episode_rewards)

            episode_states, episode_actions, episode_rewards = [],[],[]
            if mean_reward_last >= 499:
                break
if __name__ == '__main__':
    main()


            
        






