for name in dir():
    if not name.startswith('_'):
        del globals()[name]

for name in dir():
    if not name.startswith('_'):
        del locals()[name]
import gym
import tkinter
import numpy as np
#from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
import PG2 as pg
import SPSA as spsa
DISPLAY_REWARD_THRESHOLD = 300  # renders environment if total episode reward is greater then this threshold
RENDER = True  # rendering wastes time

#env = gym.make('CartPole-v0')
#env.seed(1)     # reproducible, general Policy gradient has high variance
#env = env.unwrapped


#print(self.sess.graph.collections)
#tf.reset_default_graph()
RL = pg.PolicyGradient(
    learning_rate=0.02,
    reward_decay=0.995,
     output_graph=True
     
)
#tf.reset_default_graph()
# SPSA parameters
#a = 1
#c = 1.9
#alpha = 0.602
#gamma1 = 0.101
all_rewards=[]
Num_episodes=5
avg_reward = 0
#observation = env.reset()

test_r=[]
for i_episode in range(Num_episodes):
    #tf.reset_default_graph()
    observation1 = RL.reset_env()
    action1 = RL.choose_action(observation1)
    print(action1)
    RL.store_transition(observation1, action1)
    vt = RL.learn()        

