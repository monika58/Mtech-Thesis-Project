
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

for name in dir():
    if not name.startswith('_'):
        del locals()[name]

import numpy as np
import tensorflow as tf
import SPSA as spsa
from SPSA import SimultaneousPerturbationOptimizer
import gym
import os
import threading
from gym_revisit import InGraphEnv
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
from tensorflow.core.framework import variable_pb2
import  copy
import lazy_property


# reproducible
np.random.seed(5)
tf.set_random_seed(5)



def discount_reward(r, gamma=0.99):
    discounted_ep_rs = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_ep_rs[t] = running_add
        # normalize episode reward
    mean_discounted_ep_rs = np.mean(discounted_ep_rs)
    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)
    return np.sum(discounted_ep_rs)

def _parse_shape(space):
    if isinstance(space, gym.spaces.Discrete):
        return tf.TensorShape([1, ])
    if isinstance(space, gym.spaces.Box):
        return space.shape
    raise NotImplementedError()

def _parse_dtype(space):
    if isinstance(space, gym.spaces.Discrete):
        return tf.int32
    if isinstance(space, gym.spaces.Box):
        return tf.float32
    raise NotImplementedError()

class Network(object):

    def __init__(self, session, env, name, input_shape, output_dim, optimizer, num_episodes, gym1, gym2):
        """Network structure is defined here
        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries
                TODO: create a summary op
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        # self.coord = coord
        self.name = name
        self.optimizer = optimizer
        self.num_episodes = num_episodes
        self.gym1 = gym1
        #self._env = self.env
        self.gym2 = gym2
        self.gym=self.gym1
        # self.states1=np.reshape(self.states1,[-1, self.input_shape])
        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, input_shape], name="states")
            #states = tf.placeholder(tf.float32, shape=[None, input_shape], name="states")
            # self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            #self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")

            net = self.states
            #net = states

            with tf.variable_scope("layer1"):
                net = tf.layers.dense(inputs=net, units=4, activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3,
                                                                                      dtype=tf.dtypes.float32),
                                      bias_initializer=tf.constant_initializer(0.1, dtype=tf.dtypes.float32),
                                      name='fc1')

            with tf.variable_scope("layer2"):
                net = tf.layers.dense(inputs=net, units=10, activation=None,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3,
                                                                                      dtype=tf.dtypes.float32),
                                      bias_initializer=tf.constant_initializer(0.1, dtype=tf.dtypes.float32),
                                      name='fc2')
            self.observ_shape = _parse_shape(self.env.observation_space)
            self.observ_dtype = _parse_dtype(self.env.observation_space)
            self.action_shape = _parse_shape(self.env.action_space)
            self.action_dtype = _parse_dtype(self.env.action_space)

            self.observ = tf.Variable(tf.zeros(self.observ_shape, self.observ_dtype), name='observ', trainable=False)
            self.action = tf.Variable(tf.zeros(self.action_shape, self.action_dtype), name='action', trainable=False)
            self.reward = tf.Variable(0.0, dtype=tf.float32, name='reward', trainable=False)
            self.done = tf.Variable(True, dtype=tf.bool, name='done', trainable=False)
            self.episodic_rewards = tf.Variable(0.0, dtype=tf.float32, name='episodic_rewards', trainable=False)

            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.action_prob = tf.nn.softmax(actions, name="action_prob")
            self.log_policy = tf.nn.log_softmax(actions, name='log_policy')
            #self.gym1.__init__(self.env)
            self.choose_action = tf.squeeze(tf.multinomial(self.log_policy, num_samples=1, output_dtype=tf.int32),axis=0)
            #choose_action = tf.squeeze(tf.multinomial(self.log_policy, num_samples=1, output_dtype=tf.int32), axis=0)
            #self.episodic_rewards = self.cal_episodic_reward()
            #loss1 = - tf.reduce_mean(self.episodic_rewards)
            #self.loss()
            #self.optimize()
            #self.optimizer = spsa.SimultaneousPerturbationOptimizer().minimize(self.loss1,loss2)

    def loss(self):
        self.gym=self.gym1
        self.episodic_rewards = self.cal_episodic_reward()
        loss1 = - tf.reduce_mean(self.episodic_rewards)
        self.gym = self.gym2
        self.episodic_rewards = self.cal_episodic_reward()
        loss2 = - tf.reduce_mean(self.episodic_rewards)
        return loss1,loss2

    def optimize(self):
        loss1, loss2 = self.loss()
        return spsa.SimultaneousPerturbationOptimizer().minimize(loss1,loss2)


    def choose_action1(self):
        #self.observ.set_shape(self.observ_shape)
        print("inside choose action1")
        states3 = tf.expand_dims(tf.convert_to_tensor(self.observ, dtype=None, name=None, preferred_dtype=None), 0)
        print(states3)
        states4 = self.sess.run(states3)
        #print(states4)
        feed1 = {self.states: states4}
        action = self.sess.run(self.choose_action, feed1)
        print(action)
        return action

    def loop_cond(self, observ, action, reward, done):
        #print("loop cond")
        return tf.math.logical_not(tf.convert_to_tensor(self.done))

    '''def loop_cond(observ, action, reward, done):
        return tf.math.logical_not(tf.convert_to_tensor(done))

    def loop_body(observ, action, reward, done):
        states3 = tf.expand_dims(tf.convert_to_tensor(observ, dtype=None, name=None, preferred_dtype=None), 0)
        sess=tf.get_default_session()
        states3 = sess.run(states3)
        feed = {states: states3}
        action = sess.run(choose_action, feed)
        observ, action,reward, done = gym1.simulate(action)
        return observ, action,reward, done

    def cal_episodic_reward(observ, action,reward, done,step):
        step.assign(1)
        observ, action,reward, done = tf.while_loop(cond=loop_cond, body=loop_body,loop_vars=[observ, action,reward, done])
        step.assign_add(1)
        return step
    '''

    def loop_body(self,observ1, action1, reward1, done1):
        print(observ1)
        self.observ.assign(observ1)
        print(self.observ)
        self.reward.assign(reward1)
        self.done.assign(done1)
        action1 =  self.choose_action1()
        self.action.assign(action1)
        observ2, reward2, done2 = self.gym.simulate(self.action)
        return observ2, action1, reward2, done2

    def cal_episodic_reward(self):
        self.episodic_rewards.assign(0.0)
        #self.step.assign(1)
        #done = self.done
        #self._env1 = copy.deepcopy(self._env)
        states1 = self.gym.reset()
        states1 = tf.expand_dims(tf.convert_to_tensor(states1, dtype=None, name=None, preferred_dtype=None), 0)
        self.sess.run(states1)

        #action = self.sess.partial_run_setup(self.choose_action, [states])
        print("action in episodic reward fn")
        print("now main function")
        action1 = self.choose_action
        print(action1)
        self.action.assign(action1)
        observ1, reward1, done1 = self.gym.simulate(self.action)
        print("after simulate")
        print(observ1)
        print(reward1)
        print(done1)

        self.observ.assign(observ1)
        self.reward.assign(reward1)
        self.done.assign(done1)
        self.action.assign(action1)
        '''
        #print(observ1)
        #observ1.set_shape(self.observ_shape)
        #print(observ1)
        #print(action1)
        #print(reward1)
        #print(done1)
        action1 = self.choose_action1()
        print(action1)
        #print(self.choose_action1())
        self.action.assign(action1)
        observ1, reward1, done1 = self.gym1.simulate(self.action)
        self.observ.assign(observ1)
        self.reward.assign(reward1)
        self.done.assign(done1)

        action1 = self.choose_action1()
        print(action1)
        #print(self.choose_action1())
        self.action.assign(action1)
        observ1, reward1, done1 = self.gym1.simulate(self.action)
        '''
        for i in range(50):
            self.episodic_rewards.assign_add(reward1)
            if(self.loop_cond(observ1, action1, reward1, done1)==True):
                break
            [observ1, action1, reward1, done1] = tf.contrib.eager.py_func(self.loop_body,[observ1, action1, reward1, done1],Tout=[self.observ_dtype,self.action_dtype,tf.float32,tf.bool] )


        #observ1, action1, reward1, done1 = tf.while_loop(cond=self.loop_cond, body=self.loop_body,loop_vars=[observ1, action1, reward1, done1])
        #self.step.assign_add(1)
        return self.episodic_rewards


    '''
    def train(self):
        for i in range(self.num_episodes):
            states1 = self.gym1.reset()
            print("reset")
            print(states1)
            #self.choose_action1(states1)
            states1 = tf.expand_dims(tf.convert_to_tensor(states1, dtype=None, name=None, preferred_dtype=None), 0)
            #print(states1)
            states2 = self.sess.run(states1)
            feed = {self.states: states2}
            train_op = self.sess.run(self.optimizer, feed)
    '''


def main():
    tf.reset_default_graph()
    input_shape = 4
    output_dim = 2
    env = gym.make("CartPole-v0")
    env.unwrapped
    env.seed(1)
    num_episodes = 5
    gym1 = InGraphEnv(env)
    gym2 = InGraphEnv(copy.deepcopy(env))
    optimizer = SimultaneousPerturbationOptimizer()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        single_agent = Network(env=env,
                               session=sess,
                               # coord=coord,
                               name="th",
                               input_shape=input_shape,
                               output_dim=output_dim,
                               optimizer=optimizer,
                               num_episodes=num_episodes,
                               gym1=gym1,
                               gym2=gym2
                               )
        # print("run agent")
        #init = tf.global_variables_initializer()
        sess.run(init)
        print(init)
        #single_agent.loss()
        single_agent.optimize().run()


if __name__ == '__main__':
    main()
