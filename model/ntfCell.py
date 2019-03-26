from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.rnn_cell_impl import *

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class ntfCell(LayerRNNCell):
  """DOC
  """

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, name=None, num_var = 16,#n_seg=5,
               max_vals = [], all_seg_lens = [],
                dtype=None, **kwargs):
    """Initialize the parameters for an NTF cell.
    Args:
      num_units: int, The number of units in the NTF cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
    """
    super(ntfCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer)
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple

    self._num_var = num_var

    self._num_splits = 4

    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units
    self._max_values=tf.convert_to_tensor(max_vals, dtype=tf.float32)
    self._seg_lens =tf.convert_to_tensor(all_seg_lens, dtype=tf.float32)

    self._n_seg = int(len(all_seg_lens))


  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, self._num_splits * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)

    # attention in inputs
    self._kernel_attention = self.add_variable(
        "_kernel_attention",
        shape=[self._n_seg,2*self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._bias_attention = self.add_variable(
        "_bias_attention",
        shape=[2* self._num_units],
        initializer=init_ops.zeros_initializer)


    self._kernel_context = self.add_variable(
        "traffic_context/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_var*self._n_seg],
        initializer=self._initializer,#tf.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.25),#tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
        partitioner=maybe_partitioner)
    self._bias_context = self.add_variable(
        "traffic_context/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_var*self._n_seg],
        initializer=init_ops.zeros_initializer)

    # self._kernel_context2 = self.add_variable(
    #     "traffic_context2/%s" % _WEIGHTS_VARIABLE_NAME,
    #     shape=[self._num_var*self._n_seg,self._num_var*self._n_seg],
    #     initializer=self._initializer,
    #     partitioner=maybe_partitioner)
    # self._bias_context2 = self.add_variable(
    #     "traffic_context2/%s" % _BIAS_VARIABLE_NAME,
    #     shape=[self._num_var*self._n_seg],
    #     initializer=init_ops.zeros_initializer)

    self._out_weights = self.add_variable(
        "_out_weights/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.ones_initializer,#self._initializer,#init_ops.ones_initializer,
        partitioner=maybe_partitioner)
    self._in_weights = self.add_variable(
        "_in_weights/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer,
        partitioner=maybe_partitioner)
    self._in_means = self.add_variable(
        "_in_means/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer,
        partitioner=maybe_partitioner)

    self._kernel_outm = self.add_variable(
        "traffic_outm/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_units],
        initializer=tf.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.25),
        partitioner=maybe_partitioner)
    self._bias_outm = self.add_variable(
        "traffic_outm/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer)


    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._num_splits * self._num_units],
        initializer=initializer)
    if self._use_peepholes:
      self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                         initializer=self._initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None
          else None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, must be 2-D, `[batch, input_size]`.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    input_size = inputs.get_shape().with_rank(2).dims[1].value
    if input_size is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # m_prev = tf.multiply(self._in_weights,m_prev)
    # in_gamma = self._in_weights
    # eps_gamma = tf.random_normal(shape=tf.shape(in_gamma),mean=0, stddev=1, dtype=tf.float32)
    # meas_gamma = self._in_means + tf.multiply(in_gamma,eps_gamma)

    # inputs_err = inputs #+ meas_gamma

    un_inputs = tf.multiply(inputs,self._max_values+1e-6)
    att_a = sigmoid(-(un_inputs*1e15-1e9))
    att_b = sigmoid((un_inputs*1e15-1e9))
    inputs2 = inputs*att_b + m_prev*att_a

    lstm_matrix = math_ops.matmul(inputs2, self._kernel)

    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o  = array_ops.split(
        value=lstm_matrix, num_or_size_splits=self._num_splits, axis=1)

    # Diagonal connections
    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    ntf_matrix = math_ops.matmul(m, self._kernel_context)
    ntf_matrix = sigmoid(nn_ops.bias_add(ntf_matrix, self._bias_context))#tf.nn.relu

    # ntf_matrix = tf.layers.dropout(ntf_matrix,rate=0.5)
    # ntf_matrix = math_ops.matmul(ntf_matrix, self._kernel_context2)
    # ntf_matrix = sigmoid(nn_ops.bias_add(ntf_matrix, self._bias_context2))

    # ntf_matrix = tf.Print(ntf_matrix,[ntf_matrix,tf.math.reduce_mean(ntf_matrix),tf.math.reduce_max(ntf_matrix),tf.math.reduce_min(ntf_matrix)],"ntf_matrix",summarize=10,first_n=50)

    traffic_variables = tf.reshape(ntf_matrix,[-1,self._n_seg,self._num_var])

    unscaled_inputs = tf.nn.relu(tf.multiply(inputs2,self._max_values+1e-3))
    unscaled_inputs = tf.reshape(unscaled_inputs,[-1,self._n_seg,5])
    unscaled_inputs = tf.Print(unscaled_inputs,[unscaled_inputs,tf.math.reduce_mean(unscaled_inputs),tf.math.reduce_max(unscaled_inputs),tf.math.reduce_min(unscaled_inputs)],"unscaled_inputs",summarize=10,first_n=50)

    flow_to_hr = tf.constant(120.0,name="flow_to_hr")
    flow_scaling = tf.constant(100.0,name="flow_scaling")
    density_scaling = tf.constant(100.0,name="density_scaling")

    v_f = tf.constant(240.,name="v_f") * tf.reshape(tf.reduce_mean((traffic_variables[:,:,7]),1),[-1,1])#120
    v_f =  tf.clip_by_value(v_f,50.0,120.0)
    a = tf.constant(2.86,name="a") * tf.reshape(tf.reduce_mean((traffic_variables[:,:,8]),1),[-1,1])#1.4324
    a =  tf.clip_by_value(a,0.9,1.9)
    p_cr = tf.constant(67.0,name="pcr") * tf.reshape(tf.reduce_mean((traffic_variables[:,:,9]),1),[-1,1])#33.5
    p_cr =  tf.clip_by_value(p_cr,1.0,200.0)

    lane_num = tf.constant(6.0,name="lane_num") * traffic_variables[:,:,12]#check flow_to hr
    T = tf.truediv(tf.constant(10.0,name="T"),3600.0) # 1e-6* traffic_variables[:,:,7] #check log exp v
    seg_len = tf.truediv(self._seg_lens,1000.0)#*tf.exp(traffic_variables[:,:,7])# tf.truediv(self._seg_lens,1000.0)
    tau =  tf.truediv(tf.constant(20.,name="tau"),3600.0) #* traffic_variables[:,:,8]#20.0
    nu = tf.constant(70.,name="nu") * 0.5#tf.reshape(tf.reduce_mean(traffic_variables[:,:,13],1),[-1,1])#35.0
    kappa = tf.constant(26.,name="kappa") * 0.5#tf.reshape(tf.reduce_mean(traffic_variables[:,:,14],1),[-1,1]) #13.0
    delta = tf.constant(2.8,name="delta") *0.5# tf.reshape(tf.reduce_mean(traffic_variables[:,:,15],1),[-1,1]) #1.4

    current_flows = tf.multiply(unscaled_inputs[:,:,0],flow_to_hr)
    current_velocities = unscaled_inputs[:,:,2]

    g = tf.constant(10.0,name="init_g")*tf.reshape(tf.reduce_mean((traffic_variables[:,:,6]),1),[-1,1])#traffic_variables[:,:,5]#*tf.reshape(tf.reduce_mean(traffic_variables[:,:,5],1),[-1,1])#tf.expand_dims(tf.reduce_mean(traffic_variables[:,:,5],axis=-1),-1)#0.3#0.06#
    # g =  tf.clip_by_value(g,1e-6,10.0)
    current_densities = tf.multiply(unscaled_inputs[:,:,1],g+1e-6)#+0.0*tf.truediv(current_flows, current_velocities*lane_num + 1e-6) # tf.multiply(unscaled_inputs[:,:,1], g)#tf.truediv(current_flows, current_velocities*lane_num + 1e-6) #unscaled_inputs[:,:,1] * g#tf.truediv(current_flows, current_velocities*lane_num + 1e-6)
    g = tf.Print(g,[g,tf.math.reduce_mean(g)],"g",summarize=10,first_n=10)


    r_in = tf.multiply(unscaled_inputs[:,:,3],flow_to_hr)
    r_out = tf.multiply(unscaled_inputs[:,:,4],flow_to_hr)

    first_flow     = tf.nn.relu(tf.multiply(tf.reduce_mean((traffic_variables[:,:,0]),1),flow_to_hr*flow_scaling))
    first_flow      = tf.clip_by_value(first_flow,1.0,10000.0)
    first_density  = tf.nn.relu(tf.multiply(tf.reduce_mean((traffic_variables[:,:,1]),1),density_scaling))
    first_density = tf.clip_by_value(first_density,0.1,500.0)
    first_velocity = tf.nn.relu(tf.multiply(tf.reduce_mean((traffic_variables[:,:,2]),1),240.0))
    first_velocity = tf.clip_by_value(first_velocity,30.0,120.0)
    last_density   = tf.nn.relu(tf.multiply(tf.reduce_mean((traffic_variables[:,:,3]),1),density_scaling))

    first_flow     = tf.reshape(first_flow,[-1,1])
    first_density  = tf.reshape(first_density,[-1,1])
    first_velocity = tf.reshape(first_velocity,[-1,1])
    last_density   = tf.reshape(last_density,[-1,1])

    first_flow = tf.Print(first_flow,[first_flow,tf.math.reduce_mean(first_flow)],"first_flow",summarize=10,first_n=10)
    first_density = tf.Print(first_density,[first_density,tf.math.reduce_mean(first_density)],"first_density",summarize=10,first_n=10)
    first_velocity = tf.Print(first_velocity,[first_velocity,tf.math.reduce_mean(first_velocity)],"first_velocity",summarize=10,first_n=10)
    last_density = tf.Print(last_density,[last_density,tf.math.reduce_mean(last_density)],"last_density",summarize=10,first_n=10)

    prev_flows      = tf.concat([first_flow,current_flows[:,:-1]],1)
    prev_densities  = tf.concat([first_density,current_densities[:,:-1]],1)
    prev_velocities = tf.concat([first_velocity,current_velocities[:,:-1]],1)
    next_densities  = tf.concat([current_densities[:,1:],last_density],1)

    #flow_beta * current_flows
    future_r_in = 1.*tf.nn.relu(tf.multiply(traffic_variables[:,:,4],flow_scaling))#*flow_scaling#tf.truediv(flow_scaling * traffic_variables[:,:,3],120.0)
    beta_out = tf.clip_by_value(traffic_variables[:,:,5],0.0,1.0)
    future_r_out = 1.*tf.nn.relu(tf.truediv(tf.multiply(beta_out,prev_flows),flow_to_hr))#flow_scaling#0.*traffic_variables[:,:,4]*current_flows

    #future_r_in = tf.Print(future_r_in,[future_r_in,tf.math.reduce_mean(future_r_in),tf.math.reduce_max(future_r_in),tf.math.reduce_min(future_r_in)],"future_r_in",summarize=10,first_n=50)
    #future_r_out = tf.Print(future_r_out,[future_r_out,tf.math.reduce_mean(future_r_out),tf.math.reduce_max(future_r_out),tf.math.reduce_min(future_r_out)],"future_r_out",summarize=10,first_n=50)


    """unscaled_inputs is the current_seg volume and density [32,45,2]
        prev_segs is previous timestep volume and density [32,45,2]
        next_segs is next timestep volume and density [32,45,2]
        eq_vars are the variables in equation (up to 16) [32,45,16]
    """
    v_f = tf.Print(v_f,[v_f,tf.math.reduce_mean(v_f)],"v_f",summarize=10,first_n=20)
    a = tf.Print(a,[a,tf.math.reduce_mean(a)],"a",summarize=10,first_n=20)
    p_cr = tf.Print(p_cr,[p_cr,tf.math.reduce_mean(p_cr)],"p_cr",summarize=10,first_n=20)

    with tf.name_scope("next_density"):
        future_rho =  current_densities + tf.multiply(tf.truediv(T,tf.multiply(seg_len,lane_num)),(prev_flows - current_flows + r_in - r_out))
        # future_rho =  tf.clip_by_value(future_rho,0.1,1000.0)
        future_rho =  tf.Print(future_rho,[future_rho,tf.math.reduce_max(future_rho),tf.shape(future_rho)],"future_rho",summarize=10,first_n=10)#[32 45]

    with tf.name_scope("future_velocity"):
        stat_speed =  tf.multiply( v_f, tf.exp( (tf.multiply(tf.truediv(-1.0,a),tf.math.pow(tf.truediv(current_densities,p_cr),a)))))
        stat_speed =  tf.clip_by_value(stat_speed,30.0,120.0)
        stat_speed = tf.Print(stat_speed,[stat_speed,tf.math.reduce_max(stat_speed),tf.shape(stat_speed)],"stat_speed",summarize=10,first_n=10)

        sigma_v = 20.0*(traffic_variables[:,:,10])
        # noise_v = tf.random_normal(shape=tf.shape(sigma_v),mean=0, stddev=1, dtype=tf.float32)
        epsilon_v = sigma_v#tf.multiply(sigma_v,noise_v)
        epsilon_v = tf.Print(epsilon_v,[epsilon_v,tf.math.reduce_min(epsilon_v),tf.math.reduce_max(epsilon_v)],"epsilon_v",summarize=10,first_n=10)#[32,45]

        future_vel = current_velocities + ( (T/tau) * (stat_speed - current_velocities) )\
                        + ( (current_velocities*T/seg_len) * (prev_velocities - current_velocities ) )\
                        - ( (nu*T/(tau*seg_len)) * ( (next_densities - current_densities) / (current_densities + kappa) )  )\
                        - ( (delta*T/(seg_len*lane_num)) * ( (r_in * current_velocities) / (current_densities+kappa) ) )\
                        +  1.0*epsilon_v

        future_vel = tf.Print(future_vel,[future_vel,tf.math.reduce_max(future_vel),tf.shape(future_vel)],"future_vel",summarize=10,first_n=10)#[32,45]
        future_vel = tf.clip_by_value(future_vel,30.0,120.)

    sigma_q = 200.0*(traffic_variables[:,:,11])
    # noise_q = tf.random_normal(shape=tf.shape(sigma_q),mean=0, stddev=1, dtype=tf.float32)
    epsilon_q = sigma_q#tf.multiply(sigma_q,noise_q)

    future_flows = tf.multiply(future_rho,future_vel*lane_num) + 1.0*epsilon_q
    future_flows = tf.clip_by_value(future_flows,0.0,10000.0)

    future_volumes = tf.truediv(future_flows,flow_to_hr)
    future_occupancies = tf.truediv(future_rho,g+1e-6)#tf.truediv(future_rho,g+1e-6)
    future_occupancies = tf.clip_by_value(future_occupancies,0.0,100.0)

    #future_volumes = tf.Print(future_volumes,[future_volumes,tf.math.reduce_max(future_volumes),tf.shape(future_volumes)],"future_volumes",summarize=10,first_n=10)
    #future_occupancies = tf.Print(future_occupancies,[future_occupancies,tf.math.reduce_max(future_occupancies),tf.shape(future_occupancies)],"future_occupancies",summarize=10,first_n=10)

    future_states = tf.stack([future_volumes,future_occupancies,future_vel,future_r_in,future_r_out],axis=2)

    future_states.set_shape([unscaled_inputs.get_shape()[0],self._n_seg,5])

    future_states = tf.reshape(future_states,[-1,5*self._n_seg])
    new_m = tf.truediv(future_states, (self._max_values+1e-3))

    # log_eps_out = self._out_weights
    # sample_out = tf.random_normal(shape=tf.shape(log_eps_out),mean=0, stddev=1, dtype=tf.float32)
    # epsilon_out = tf.multiply(tf.exp(log_eps_out),sample_out)
    #
    # m = new_m + epsilon_out#tf.multiply(self._out_weights,new_m)#new_m
    m = tf.clip_by_value(new_m,0.0,100.0) #- meas_gamma#tf.multiply(self._out_weights,new_m)#new_m

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))

    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(ntfCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))




#####LSTMCell2 delete later from here
class LSTMCell2(LayerRNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf
  Felix Gers, Jurgen Schmidhuber, and Fred Cummins.
  "Learning to forget: Continual prediction with LSTM." IET, 850-855, 1999.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, name=None, dtype=None, **kwargs):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
      When restoring from CudnnLSTM-trained checkpoints, use
      `CudnnCompatibleLSTMCell2` instead.
    """
    super(LSTMCell2, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer)
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple

    self._num_splits = 4

    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape= [self._num_splits * self._num_units],
        initializer=initializer)
    if self._use_peepholes:
      self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                         initializer=self._initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None
          else None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, must be 2-D, `[batch, input_size]`.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    # o = tf.Print(o,[o,tf.shape(o)],"o")
    # Diagonal connections
    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      m = math_ops.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    # m = tf.Print(m,[m,tf.shape(m)],"m")

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(LSTMCell2, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
