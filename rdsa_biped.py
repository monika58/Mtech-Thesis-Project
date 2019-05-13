import numpy as np
import pickle
import copy
import gym
import matplotlib.pyplot as plt
import time

def tanh(x):
    #f = np.exp(x - np.max(x))  # shift values
    return np.tanh(x)


def sigmoid(inpt):
    return 1.0 / (1 + np.exp(-1 * inpt))


def leaky_relu(x):
    #result = inpt
    #result[inpt < 0] = 0.01*inpt
    leaky_way1 = np.where(x > 0, x, x * 0.01)
    return leaky_way1

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result


def softmax(x):
    f = np.exp(x - np.max(x))  # shift values
    return f / f.sum(axis=0)


def normalize(X):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X


def train_network(weights, envv, activation):
    episodic_rewards = 0
    data_inputs = envv.reset()
    state = data_inputs
    done = False
    t = 0
    while not done:
        r1 = []
        r1 = np.expand_dims(np.append(normalize(state),1), axis=0)
        for idx in range(len(weights) - 1):
            curr_weights = weights[idx]
            r1 = np.matmul(r1, curr_weights)
            r1 = drop_out(0.5,r1)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
            elif activation == "tanh":
                r1 = tanh(r1)
        curr_weights = weights[-1]
        r1 = np.matmul(r1, curr_weights)
        r1 = tanh(r1[0])
        #predicted_label = np.clip(np.random.normal(r1, variance), -1, 1)
        #print(r1)
        next_state, reward, done, info = envv.step(r1)
        state = next_state
        episodic_rewards = discount_factor*episodic_rewards + reward
        t=t+1
    return episodic_rewards,t


def delta(var, algo, epsilon):
    delta1 = []
    for i in range(len(var)):
        if algo == "uniform":
            d = 2 * np.round(np.random.uniform(0,1,(var[i].shape[0], var[i].shape[1]))) - 1
            delta1.append(d)
        else:
            d= np.random.uniform(0,1,(var[i].shape[0], var[i].shape[1]))
            for k in range(len(d)):
                for j in range(len(d[k])):
                    if d[k][j]<((1+epsilon)/(2+epsilon)):
                        d[k][j]=-1
                    else:
                        d[k][j]=1+epsilon
            delta1.append(d)
    return delta1


def perturb_weights(var, delta2, ck):
    var1 = []
    for i in range(len(var)):
        v = var[i] + ck * delta2[i]
        var1.append(v)
    return var1

'''
def soft_updates(delta2, ck, diff):
    g = []
    for i in range(len(delta2)):
        g1=diff / (2 * ck * delta2[i])
        g.append(g1)
    return g
'''

def update_weights(var, delta2, ck, diff, epsilon, algo):
    var2 = []
    for i in range(len(var)):
        if algo == "uniform" :
            v = var[i] + 3 * diff * delta2[i] / (2 * ck )
        else:
            v = var[i] + diff*delta2[i]/ (2 * ck *(1+epsilon))
        var2.append(v)
    return var2


def weight_initialization(input_shape,HL1_neurons, output_neurons, init_type):
    input_HL1_weights=[]
    HL2_output_weights = []
    if (init_type == "xavier"):
        input_HL1_weights = np.random.randn(input_shape, HL1_neurons)/np.sqrt(input_shape / 2.)
        HL2_output_weights = np.random.randn(HL1_neurons, output_neurons)/np.sqrt(HL1_neurons / 2.)
        #weights = np.array([input_HL1_weights, HL2_output_weights])
    elif(init_type =="uniform"):
        input_HL1_weights = np.random.uniform(low=-0.1, high=0.1,
                                              size=(input_shape, HL1_neurons))/np.sqrt(input_shape / 2.)
        HL2_output_weights = np.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, output_neurons))/np.sqrt(HL1_neurons / 2.)
    weights = np.array([input_HL1_weights, HL2_output_weights])
    #b1 = np.zeros((1, H))
    #b2 = np.zeros((1, H))
    return weights

def drop_out(p,h1):
    u1 = np.random.binomial(1, p, size=h1.shape)
    h1 *= u1
    return h1


final_weights = []
final_rewards = []
replications = 5
env = gym.make('BipedalWalker-v2')
input_shape = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
env.seed(1)
algo = "nonuniform"
epsilon = 0.01
episodic_length = []
for j in range(replications):
    #np.random.seed(0)
    env1 = copy.deepcopy(env)
    env2 = copy.deepcopy(env)
    env3 = copy.deepcopy(env)
    # num_episodes = 5
    # alpha = 0.602
    # gamma = 0.101
    # a = 0.16
    # c = 1
    # A = 100
    a = 1.0
    c = 1.9
    A = 50
    N = 1000
    alpha = 1
    gamma = 1 / 6
    seed = 5
    discount_factor = 0.995
    HL1_neurons = 50
    weights = weight_initialization(input_shape+1, HL1_neurons, output_dim, "uniform")
    #print("weights= ", weights)
    rewards = []
    rewards_plus = []
    rewards_minus = []
    mean_rewards = []
    for i in range(N):
        ak = a / (i + 1+A) ** alpha
        ck = c / (i + 1) ** gamma
        delta2 = delta(weights,algo,epsilon)
        weights_plus = perturb_weights(weights, delta2, ck)
        weights_minus = perturb_weights(weights, delta2, -ck)
        rewplus, tplus = train_network(weights_plus, env1, activation="tanh")
        rewminus, tminus = train_network(weights_minus, env2, activation="tanh")
        rewards1, t = train_network(weights, env3, activation="tanh")
        rewards.append(rewards1)
        episodic_length.append(t)
        m = np.mean(rewards[:-10])
        if m >195:
            break
        mean_rewards.append(m)
        rewards_plus.append(rewplus)
        rewards_minus.append(rewminus)
        print('rep = ',j,'episode =' ,i, 'reward = ',rewards1)
        diff = (rewplus - rewminus) * ak
        weights = update_weights(weights, delta2, ck, diff,epsilon, algo)
    final_rewards.append(rewards)


np.save('rdsa_asymb_biped_rew1',final_rewards)













