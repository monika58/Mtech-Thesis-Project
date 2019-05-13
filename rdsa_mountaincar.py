import pickle
import copy
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from tqdm import tqdm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor
variance = 0.01
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



def tanh(x):
    #f = np.exp(x - np.max(x))  # shift values
    return np.tanh(x)



def normalize(X):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X


def train_network(weights, envv, activation="relu", rendering=False):
    pos =[]
    vel = []
    rew = []
    episodic_rewards = 0
    data_inputs = envv.reset()
    state = data_inputs
    done = False
    t=0
    while not done:
        pos.append(state[0])
        vel.append(state[1])
        r1 = []
        r1 = np.expand_dims(np.append(normalize(state),1), axis=0)
        for idx in range(len(weights) - 1):
            curr_weights = weights[idx]
            r1 = np.matmul(r1, curr_weights)
            r1 = drop_out(0.5,r1)
            if activation == "relu":
                r1 = leaky_relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        curr_weights = weights[-1]
        r1 = np.matmul(r1, curr_weights)
        #r1 = tanh(r1[0])

        #predicted_label = r1
        predicted_label = np.clip(np.random.normal(r1[0], variance), -1, 1)
        #print(r1)
        next_state, reward, done, info = envv.step(predicted_label)
        state = next_state
        episodic_rewards = episodic_rewards + reward
        t=t+1
        rew.append(reward)
    return episodic_rewards,t,pos,vel,rew

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
final_weights = []
final_rewards = []
rep = 1
epsilon =0.01
algo = "nonuniform" #nonuniform for asymber bern , uniform for uniform bern
for i in range(rep):
    alpha = 1
    gamma = 1 / 6
    a = 2.0
    c = 2.5
    A = 50
    N = 5000
    seed = 5
    discount_factor = 0.995
    HL1_neurons = 100
    weights = weight_initialization(input_shape+1,HL1_neurons,output_dim,"xavier")
    rewards=[]
    rewards_plus = []
    rewards_minus = []
    mean_rewards = []
    episodic_length =[]
    env1 = copy.deepcopy(env)
    env2 = copy.deepcopy(env)
    env3 = copy.deepcopy(env)
    pos_f = []
    vel_f = []
    rew_f = []
    episode_id=1
    env3 = gym.wrappers.Monitor(env3, './mc_video/',resume=True, video_callable=lambda x: episode_id%100 ==True ,force=True)
    for i in range(N):
    #print(i)

        ak = a / (i + 1+ A) ** alpha
        ck = c / (i + 1) ** gamma
        delta2 = delta(weights,algo,epsilon)
        weights_plus = perturb_weights(weights, delta2, ck)
        weights_minus = perturb_weights(weights, delta2, -ck)
        rewplus,tplus,pos1,vel1,rew1 = train_network(weights_plus, env1, activation="relu", rendering=False)
        rewminus,tminus,pos2,vel2,rew2 = train_network(weights_minus, env2, activation="relu",rendering=False)
        rewards1, t, pos3, vel3, rew3 = train_network(weights, env3, activation="relu", rendering=False)
        episodic_length.append(t)
        m = np.mean(rewards[:-10])
        if m >195:
            break
        mean_rewards.append(m)
        rewards_plus.append(rewplus)
        rewards_minus.append(rewminus)
        print('episode =' ,i, 'reward = ',rewards1)
        diff = (rewplus - rewminus) * ak
        weights = update_weights(weights, delta2, ck, diff,epsilon, algo)
    final_rewards.append(rewards)
    env3.close()


np.save('mc_wgts',weights)
np.save('rdsa_ab_1_mountaincar_rew',final_rewards)
np.save('rdsa_ab_1_mountaincar_pos',pos_f)
np.save('rdsa_ab_1_mountaincar_vel',vel_f)
np.save('rdsa_ab_1_mountaincar_rew1',rew_f)



weights = np.load('mc_wgts.npy')
env = gym.make('MountainCarContinuous-v0')
input_shape = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
env.seed(2)
episode_id = 1
env = gym.wrappers.Monitor(env, './mc_video/', resume=True, video_callable=lambda episode_id:True, force=True)
data_inputs = env.reset()
state = data_inputs
done = False
t = 0
episodic_rewards = 0
activation ="relu"
for i in range(10):
    while not done:
        env.render(mode='rgb_array')
        r1 = []
        r1 = np.expand_dims(np.append(normalize(state), 1), axis=0)
        for idx in range(len(weights) - 1):
            curr_weights = weights[idx]
            r1 = np.matmul(r1, curr_weights)
            r1 = drop_out(0.5, r1)
            if activation == "relu":
                r1 = leaky_relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        curr_weights = weights[-1]
        r1 = np.matmul(r1, curr_weights)
        r1 = tanh(r1[0])
        predicted_label = r1
    # print(r1)
        next_state, reward, done, info = env.step(predicted_label)
        state = next_state
        episodic_rewards = episodic_rewards + reward
        t = t + 1
env.close()









pos_f = np.load('rdsa_ab_1_mountaincar_pos.npy')
vel_f = np.load('rdsa_ab_1_mountaincar_pos.npy')
rew_f = np.load('rdsa_ab_1_mountaincar_rew1.npy')

X=pos_f[100]
Y=vel_f[100]
Z=rew_f[100]
plt.plot(X,Y)
for i in range(1,len(pos_f)):
    X.extend(pos_f[i])
    Y.extend(vel_f[i])
    Z.extend(rew_f[i])

X=np.array(X)
Y=np.array(Y)
Z=np.array(Z)

POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
grid_size = 40
positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)
axis_x = []
axis_y = []
axis_z = []
for position in positions:
    for velocity in velocities:
        axis_x.append(position)
        axis_y.append(velocity)
        axis_z.append(value_function.cost_to_go(position, velocity))


ax.scatter(X, Y, Z)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Cost to go')
ax.set_title('Episode %d' % (episode + 1))
