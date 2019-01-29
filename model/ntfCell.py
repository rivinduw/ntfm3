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
               activation=None, reuse=None, name=None, num_var = 11,#n_seg=5,
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

    self._num_splits = 5

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
    #  max_vals = [ 4462.11559904,    53.7349221 ,  4423.03294541,    73.53982445]
    self._max_values=tf.convert_to_tensor(max_vals, dtype=tf.float32)

    # all_seg_lens = [900.0,600.0,900.0,700.0,1800.0]
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
        shape=[input_depth + h_depth, self._num_splits * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)

    # attention in inputs
    self._kernel_attention = self.add_variable(
        "_kernel_attention",
        shape=[self._num_units,2*self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._bias_attention = self.add_variable(
        "_bias_attention",
        shape=[2 * self._num_units],
        initializer=init_ops.zeros_initializer)


    self._kernel_context = self.add_variable(
        "traffic_context/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, 4 + self._num_var*self._n_seg],# * self._n_seg
        initializer=self._initializer,#tf.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.25),#tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
        partitioner=maybe_partitioner)
    self._bias_context = self.add_variable(
        "traffic_context/%s" % _BIAS_VARIABLE_NAME,
        shape=[4 + self._num_var*self._n_seg],# * self._n_seg
        initializer=init_ops.zeros_initializer)

    self._kernel_outm = self.add_variable(
        "traffic_outm/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[2*self._n_seg, self._num_units],
        initializer=self._initializer,#tf.keras.initializers.Identity(gain=1.0),#self._initializer,#tf.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.25),#tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
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

    inputs_scaled = tf.div((inputs+1e-12),(self._max_values+1e-6))
    m_prev_scaled = tf.div((m_prev+1e-12),(self._max_values+1e-6))

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs_scaled, m_prev_scaled], 1), self._kernel) #128+90 going in, 5*128 coming out [input_depth + h_depth, 5 * self._num_units]
        # self._kernel = self.add_variable(
            # _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + h_depth, 5 * self._num_units],
            # initializer=self._initializer,
            # partitioner=maybe_partitioner)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o, att_c  = array_ops.split(
        value=lstm_matrix, num_or_size_splits=self._num_splits, axis=1)
    # o = tf.Print(o,[o,tf.shape(o)],"o")#[32 128]
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
    ntf_matrix = sigmoid(nn_ops.bias_add(ntf_matrix, self._bias_context))#)

    boundry, traffic_variables = array_ops.split(value=ntf_matrix, num_or_size_splits=[4,self._num_var*self._n_seg], axis=1)#*self._n_seg
    # boundry = tf.reshape(boundry,[-1,2,2])#32,2,2
    # boundry = tf.multiply(boundry,[20000.,500.])#tf.multiply(boundry,[tf.constant(self._max_values[0]),tf.constant(self._max_values[1])])
    # boundry = tf.Print(boundry,[boundry,tf.math.reduce_mean(boundry)],"boundry",summarize=10,first_n=50)
    # contexts = ntf_matrix#array_ops.split(value=ntf_matrix, num_or_size_splits=self._n_seg, axis=1)

    traffic_variables = tf.reshape(traffic_variables,[-1,self._n_seg,self._num_var])

    #attention
    # att_c = math_ops.matmul(att_c, self._kernel_attention)
    # att_c = nn_ops.bias_add(att_c, self._bias_attention)
    # att_c = tf.reshape(att_c,[-1,self._num_units,2])
    # att_c = att_c[:,:,0]
    # att_c = tf.nn.softmax(att_c,axis=-1) #-1 is actually default
    #
    # att_c = tf.Print(att_c,[attatt_b_c,tf.shape(att_c)],"att_c",summarize=10,first_n=10)
    m_prev= tf.Print(m_prev,[m_prev,tf.shape(m_prev)],"m_prev",summarize=10,first_n=10)
    inputs = tf.Print(inputs,[inputs,tf.shape(inputs)],"inputs",summarize=10,first_n=10)
    # # self._max_values = tf.Print(self._max_values,[self._max_values,tf.shape(self._max_values)],"_max_values",summarize=10,first_n=10)
    #
    # stacked_inputs = tf.stack([inputs, m_prev], axis=2)
    # stacked_inputs = tf.Print(stacked_inputs,[stacked_inputs,tf.shape(stacked_inputs)],"stacked_inputs",summarize=10,first_n=10)
    # unscaled_inputs = tf.reduce_sum(tf.multiply(stacked_inputs,att_c),axis=2)

    #residual add tanh
    # att_c = tf.clip_by_value(att_c,-1.0,1.0)
    # att_c = sigmoid(-inputs+3.0)#tf.math.tanh
    # att_c = tf.Print(att_c,[att_c,tf.shape(att_c)],"att_c",summarize=10,first_n=10)
    # att_b = tf.math.tanh(inputs)
    # att_b = tf.reshape(traffic_variables[:,:,5:5+5],[-1,60])
    # att_b = sigmoid(-att_b-3.0)
    # att_b = tf.Print(att_b,[att_b,tf.shape(att_b)],"att_b",summarize=10,first_n=10)

    unscaled_inputs = inputs #+ tf.multiply(att_b,m_prev)
    # unscaled_inputs = inputs+ tf.reduce_mean(att_c)*0 + 0*m_prev#*(self._max_values+1e-6)
    unscaled_inputs = tf.Print(unscaled_inputs,[unscaled_inputs,tf.shape(unscaled_inputs)],"unscaled_inputs1",summarize=10,first_n=10)
    unscaled_inputs = tf.reshape(unscaled_inputs,[-1,self._n_seg,5])#32,45,2

    #Fixed variables
    T = tf.constant(10.0,name="T")
    seg_len = self._seg_lens
    tau = tf.constant(12.,name="tau")
    nu = tf.constant(35.,name="nu")
    kappa = tf.constant(13.,name="kappa")
    delta = tf.constant(1.4,name="delta")


    flow_scaling = tf.constant(12000.0,name="flow_scaling") #from (0,1) to..
    density_scaling = tf.constant(40.0,name="density_scaling")

    v_f = tf.constant(200.,name="v_f") * traffic_variables[:,:,0]
    a = tf.constant(3.0,name="a") * traffic_variables[:,:,1]
    p_cr = tf.constant(1000.0,name="pcr") * traffic_variables[:,:,2]

    future_r_in = flow_scaling * traffic_variables[:,:,3]
    future_r_out = flow_scaling * traffic_variables[:,:,4]

    g = tf.constant(10.0,name="g") * traffic_variables[:,:,5]

    lane_num = tf.constant(6.0,name="lane_num") * traffic_variables[:,:,6]
    lane_num = tf.Print(lane_num,[lane_num,tf.math.reduce_max(lane_num),tf.shape(lane_num)],"lane_num",summarize=10,first_n=10)
    seg_len = tf.Print(seg_len,[seg_len,tf.math.reduce_max(seg_len),tf.shape(seg_len)],"seg_len",summarize=10,first_n=10)


    current_flows = tf.multiply(unscaled_inputs[:,:,0],120.0)
    current_velocities = unscaled_inputs[:,:,2]

    current_densities =  unscaled_inputs[:,:,1] * g#tf.div(current_flows, current_velocities*lane_num + 1e-6)

    r_in = tf.multiply(unscaled_inputs[:,:,3],120.0)
    r_out = tf.multiply(unscaled_inputs[:,:,4],120.0)

    first_flow     = boundry[:,0:1]*flow_scaling #current_flows[:,:1] #: variable
    first_density  = boundry[:,1:2]*density_scaling #current_densities[:,:1] #: variable
    first_velocity = tf.div(first_flow,first_density*lane_num[:,0:1]) #lane_num is [batch_size,12]
    last_density   = boundry[:,2:3]*density_scaling #current_densities[:,-1:] #: variable

    prev_flows      = tf.concat([first_flow,current_flows[:,:-1]],1)
    prev_densities  = tf.concat([first_density,current_densities[:,:-1]],1)
    prev_velocities = tf.concat([first_velocity,current_velocities[:,:-1]],1)
    next_densities  = tf.concat([current_densities[:,1:],last_density],1)

    """unscaled_inputs is the current_seg volume and density [32,45,2]
        prev_segs is previous timestep volume and density [32,45,2]
        next_segs is next timestep volume and density [32,45,2]
        eq_vars are the variables in equation (up to 16) [32,45,16]
    """



    v_f = tf.Print(v_f,[v_f,tf.math.reduce_mean(v_f)],"v_f",summarize=10,first_n=10)
    a = tf.Print(a,[a,tf.math.reduce_mean(a)],"a",summarize=10,first_n=10)
    p_cr = tf.Print(p_cr,[p_cr,tf.math.reduce_mean(p_cr)],"p_cr",summarize=10,first_n=10)

    with tf.name_scope("next_density"):
        future_rho =  current_densities + tf.multiply(tf.div(T,tf.multiply(seg_len,lane_num)),(prev_flows - current_flows + r_in - r_out))
        future_rho = tf.Print(future_rho,[future_rho,tf.math.reduce_max(future_rho),tf.shape(future_rho)],"future_rho",summarize=10,first_n=10)#[32 45]

    with tf.name_scope("future_velocity"):
        stat_speed =  tf.multiply( v_f, tf.exp( (tf.multiply(tf.div(-1.0,a),tf.math.pow(tf.divide(current_densities,p_cr),a)))))
        stat_speed = tf.Print(stat_speed,[stat_speed,tf.math.reduce_max(stat_speed),tf.shape(stat_speed)],"stat_speed",summarize=10,first_n=10)

        future_vel = current_velocities + ( (T/tau) * (stat_speed - current_velocities) )\
                        + ( (current_velocities*T/seg_len) * (prev_velocities - current_velocities ) )\
                        - ( (nu*T/(tau*seg_len)) * ( (next_densities - current_densities) / (current_densities + kappa) )  )\
                        - ( (delta*T/(seg_len*lane_num)) * ( (r_in * current_velocities) / (current_densities+kappa) ) )
        future_vel = tf.Print(future_vel,[future_vel,tf.math.reduce_max(future_vel),tf.shape(future_vel)],"future_vel",summarize=10,first_n=10)#[32,45]
        # future_vel = tf.clip_by_value(future_vel,0.0,150.0)


    future_flows = tf.multiply(future_rho,future_vel*lane_num)#tf.divide(tf.multiply(future_rho,future_vel),tf.constant(4.0))

    future_volumes = tf.divide(future_flows,120.0)
    future_occupancies = tf.divide(future_rho,g+1e-6)

    future_states = tf.stack([future_volumes,future_occupancies,future_vel,future_r_in,future_r_out],axis=2)

    future_states = tf.reshape(future_states,[-1,5*self._n_seg])

    m = tf.nn.relu(future_states) #/ (self._max_values)

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
