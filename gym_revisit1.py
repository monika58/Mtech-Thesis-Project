from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
import  copy


class InGraphEnv():
  """Put an OpenAI Gym environment into the TensorFlow graph.
  The environment will be stepped and reset inside of the graph using
  tf.py_func(). The current observation, action, reward, and done flag are held
  in according variables.
  """

  def __init__(self, _env, graph = None,use_locking = True, name = "gym_call"):
   # super(InGraphEnv, self).__init__(use_locking, name)
    """Put an OpenAI Gym environment into the TensorFlow graph.
    Args:
      env: OpenAI Gym environment.
    """
    self._env = _env
    self.work_graph = tf.get_default_graph() if graph is None else graph
    observ_shape = self._parse_shape(self._env.observation_space)
    observ_dtype = self._parse_dtype(self._env.observation_space)
    action_shape = self._parse_shape(self._env.action_space)
    action_dtype = self._parse_dtype(self._env.action_space)
    with tf.name_scope('environment'):
      self._observ = tf.Variable(
          tf.zeros(observ_shape, observ_dtype), name='observ', trainable=False)
      self._action = tf.Variable(
          tf.zeros(action_shape, action_dtype), name='action', trainable=False)
      self._reward = tf.Variable(
          0.0, dtype=tf.float32, name='reward', trainable=False)
      self._done = tf.Variable(
          True, dtype=tf.bool, name='done', trainable=False)
      self._step = tf.Variable(
          0, dtype=tf.int32, name='step', trainable=False)
      self._episodic_reward = tf.Variable(
          0.0, dtype=tf.float32, name='episodic_reward', trainable=False)


  def _clone_model_gym(self, model, newinp, dst_scope):
      ''' make a copy of model and connect the resulting sub-graph to
          input ops of the original graph and parameter assignments by
          perturbator.
      '''
      def not_placeholder_or_trainvar_filter(op):
          if op.type == 'Placeholder':  # evaluation sub-graphs will be fed from original placeholders
              return False
          return True

      ops_without_inputs = ge.filter_ops(model.ops, not_placeholder_or_trainvar_filter)
      # remove init op from clone if already present
      try:
          ops_without_inputs.remove(self.work_graph.get_operation_by_name("init"))
      except:
          pass
      clone_sgv = ge.make_view(ops_without_inputs)
      clone_sgv = clone_sgv.remove_unused_ops(control_inputs=True)

      input_replacements = {}
      for t in clone_sgv.inputs:
          if t.name in newinp.keys():  # input from trainable var --> replace with perturbation
              input_replacements[t] = newinp[t.name]
          else:  # otherwise take input from original graph
              input_replacements[t] = self.work_graph.get_tensor_by_name(t.name)
      return ge.copy_with_input_replacements(clone_sgv, input_replacements, dst_scope=dst_scope)

  def loop_cond(self, observ,reward,done,action):
      return tf.cond(tf.math.logical_not(tf.convert_to_tensor(done)),self.loop_body,False)

  def loop_body(self, observ,reward,done,action):
      observ_dtype = self._parse_dtype(self._env.observation_space)
      observ_shape = self._parse_shape(self._env.observation_space)
      action_shape = self._parse_shape(self._env.action_space)
      action_dtype = self._parse_dtype(self._env.action_space)
      #print(observ_dtype)
      newinp = {}
      obs = tf.convert_to_tensor(observ, dtype=observ_dtype,  name='observ_modified')
      print(obs)
      newinp['states:0'] = tf.expand_dims(obs, 0)
      print(newinp)
      print("new inp done")
      _, info = self._clone_model_gym(ge.sgv(self.work_graph), newinp, 'full_episode')
      print("entered deeper")
      print([info.transformed(action)])

      #print(action)
      #action = tf.check_numerics(action, 'action')
      #print(action)
      action.set_shape(action_shape)
      print(action)
      #self.reset()
      observ, reward, done = tf.py_func(lambda a: self._env.step(a)[:3], [info.transformed(action)], [observ_dtype, tf.float32, tf.bool], name='step')
      #observ, reward, done = tf.py_func(self._env.step(y)[:3], y, [observ_dtype, tf.float32, tf.bool], name='step')
      print("ran")
      observ = tf.check_numerics(observ, 'observ')
      reward = tf.check_numerics(reward, 'reward')

      print("checked observ ")
      with tf.control_dependencies([self._observ.assign(observ), self._reward.assign(reward), self._done.assign(done)]):
          self._action.assign(y)
          self._episodic_reward.assign_add(reward)
      print("updated episodic reward ")
      observ.set_shape(observ_shape)
      #action.set_shape(action_shape)
      print(reward.shape)
      print(done.shape)
      print(action)
      #print(observ.dtype)
      return observ, reward, done, self._episodic_reward


  def simulate(self, action):
    """Step the environment.
    The result of the step can be accessed from the variables defined below.
    Args:
      action: Tensor holding the action to apply.
    Returns:
      Operation.
    """
    self._env1 = copy.deepcopy(self._env)
    with tf.name_scope('environment/simulate'):
      observ_shape = self._parse_shape(self._env.observation_space)
      observ_dtype = self._parse_dtype(self._env.observation_space)
      action_shape = self._parse_shape(self._env.action_space)
      action_dtype = self._parse_dtype(self._env.action_space)
      if action.dtype in (tf.float16, tf.float32, tf.float64):
        action = tf.check_numerics(action, 'action')
      observ_dtype = self._parse_dtype(self._env.observation_space)
      #self.reset()
      print(action)
      observ, reward, done = tf.py_func(
          lambda a: self._env.step(a)[:3], [action], [observ_dtype, tf.float32, tf.bool], name='step')
      observ = tf.check_numerics(observ, 'observ')
      reward = tf.check_numerics(reward, 'reward')
      #episodic_reward = tf.check_numerics(episodic_reward, 'episodic_reward')
      with tf.control_dependencies([self._observ.assign(observ), self._reward.assign(reward), self._done.assign(done)]):
          self._episodic_reward.assign_add(reward)
      #self._episodic_reward.assign_add(reward)
      #work_graph =  tf.get_default_graph()
      observ.set_shape(observ_shape)
      action.set_shape(action_shape)
      print(reward.shape)
      print(done.shape)

      #print("_episodic_reward")
      #print(self._episodic_reward)
      '''
      while not tf.cond(self._done):
          newinp = {}
          newinp['states:0'] = tf.expand_dims(tf.convert_to_tensor(self._observ, dtype=None, name=None,preferred_dtype=None),0)
          _, info = self._clone_model_gym(ge.sgv(self.work_graph), newinp, 'full_episode')
          observ, reward, done = tf.py_func(lambda a: self._env.step(a)[:3], [info.transformed(action)], [observ_dtype, tf.float32, tf.bool], name='step')
          self._episodic_reward.assign_add(self._episodic_reward,reward)
      '''
      observ, reward, done,episodic_reward = tf.scan(self.loop_cond, elems=[observ,reward,done,action],initializer=[observ,reward,done,action])

    return episodic_reward


  def reset(self):
    """Reset the environment.
    Returns:
      Tensor of the current observation.
    """
    observ_dtype = self._parse_dtype(self._env.observation_space)
    observ = tf.py_func(self._env.reset, [], observ_dtype, name='reset')
    observ = tf.check_numerics(observ, 'observ')
    with tf.control_dependencies([
        self._observ.assign(observ),
        self._reward.assign(0),
        self._done.assign(False),
        self._episodic_reward.assign(0)]):
      return self._observ

  #tf.identity(observ)
  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  @property
  def action(self):
    """Access the variable holding the last received action."""
    return self._action

  @property
  def reward(self):
    """Access the variable holding the current reward."""
    return self._reward

  @property
  def done(self):
    """Access the variable indicating whether the episode is done."""
    return self._done

  @property
  def step(self):
    """Access the variable containing total steps of this environment."""
    return self._step

  @property
  def episodic_reward(self):
    """Access the variable containing total steps of this environment."""
    return self._episodic_reward



  def _parse_shape(self, space):
    """Get a tensor shape from a OpenAI Gym space.
    Args:
      space: Gym space.
    Raises:
      NotImplementedError: For spaces other than Box and Discrete.
    Returns:
      Shape tuple.
    """
    if isinstance(space, gym.spaces.Discrete):
      return tf.TensorShape([1,])
    if isinstance(space, gym.spaces.Box):
      return space.shape
    raise NotImplementedError()

  def _parse_dtype(self, space):
    """Get a tensor dtype from a OpenAI Gym space.
    Args:
      space: Gym space.
    Raises:
      NotImplementedError: For spaces other than Box and Discrete.
    Returns:
      TensorFlow data type.
    """
    if isinstance(space, gym.spaces.Discrete):
      return tf.int32
    if isinstance(space, gym.spaces.Box):
      return tf.float32
    raise NotImplementedError()