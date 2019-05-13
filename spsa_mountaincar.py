import numpy as np
import pickle
import copy
import gym
import matplotlib.pyplot as plt

def sigmoid(inpt):
    return 1.0 / (1 + np.exp(-1 * inpt))


def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result


def leaky_relu(x):
    #result = inpt
    #result[inpt < 0] = 0.01*inpt
    leaky_way1 = np.where(x > 0, x, x * 0.01)
    return leaky_way1

def softmax(x):
    f = np.exp(x - np.max(x))  # shift values
    return f / f.sum(axis=0)



def tanh(x):
    #f = np.exp(x - np.max(x))  # shift values
    return np.tanh(x)



def normalize(X):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X


def train_network(weights, envv, activation="relu"):
    episodic_rewards = 0
    data_inputs = envv.reset()
    # print(data_inputs)
    state = data_inputs
    #state -= np.mean(state, axis=0)
    #state /= np.std(state, axis=0)
    #state = normalize(state)
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

def delta(var):
    delta1 = []
    for i in range(len(var)):
        d = 2 * np.round(np.random.rand(var[i].shape[0], var[i].shape[1])) - 1
        delta1.append(d)
    return delta1


def perturb_weights(var, delta2, ck):
    var1 = []
    for i in range(len(var)):
        v = var[i] + ck * delta2[i]
        var1.append(v)
    return var1


def update_weights(var, delta2, ck, diff):
    var2 = []
    for i in range(len(var)):
        v = var[i] + diff / (2 * ck * delta2[i])
        var2.append(v)
    return var2


def weight_initialization(input_shape,HL1_neurons,output_neurons, init_type):
    input_HL1_weights=[]
    #HL1_HL2_weights = []
    HL2_output_weights = []
    if (init_type == "xavier"):
        input_HL1_weights = np.random.randn(input_shape, HL1_neurons)/np.sqrt(input_shape / 2.)
        #HL1_HL2_weights = np.random.randn(HL1_neurons, HL2_neurons) / np.sqrt(HL1_neurons / 2.)
        HL2_output_weights = np.random.randn(HL1_neurons, output_neurons)/np.sqrt(HL1_neurons / 2.)
        #weights = np.array([input_HL1_weights, HL2_output_weights])
        b1 = np.ones(input_shape)
        b2 = np.ones(HL1_neurons)
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

env = gym.make('MountainCarContinuous-v0')
input_shape = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
env.seed(1)
#num_episodes = 5
#a=0.1
#c=0.1
#alpha = 1.0
#gamma =0.4
alpha = 1
gamma = 1/6
a = 1.0
c = 1.9
A = 50
N = 2000
seed = 5
discount_factor = 0.995
variance = 3
HL1_neurons = 50
weights = weight_initialization(input_shape+1,HL1_neurons,output_dim,"uniform")

rewards=[]
rewards_plus = []
rewards_minus = []
mean_rewards = []
episodic_length =[]

for i in range(N):
    if variance >= 0.1:
        variance *= .985
    ak = a / (i + 1) ** alpha
    ck = c / (i + 1) ** gamma
    print('ak =', ak, 'ck = ', ck)
    delta2 = delta(weights)
    weights_plus = perturb_weights(weights, delta2, ck)
    weights_minus = perturb_weights(weights, delta2, -ck)
    env1 = copy.deepcopy(env)
    env2= copy.deepcopy(env)
    rewplus,tplus = train_network(weights_plus, env, activation="tanh")
    rewminus,tminus = train_network(weights_minus, env1, activation="tanh")
    rewards1,t = train_network(weights, env2, activation="tanh")
    rewards.append(rewards1)
    episodic_length.append(t)
    m = np.mean(rewards[:-10])
    mean_rewards.append(m)
    rewards_plus.append(rewplus)
    rewards_minus.append(rewminus)
    print('episode =', i, 'reward = ', rewards1)
    diff = (rewplus - rewminus) * ak
    weights = update_weights(weights, delta2, ck, diff)

smoothed_rewards =[]
for k in range(len(final_rewards)):
    sr = [np.mean(final_rewards[k][max(0,i-50):i+1]) for i in range(len(final_rewards[k]))]
    std = [np.std(final_rewards[k][max(0,i-50):i+1]) for i in range(len(final_rewards[k]))]
    smoothed_rewards.append(sr)


