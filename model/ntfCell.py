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
        shape=[input_depth, self._num_splits * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    # self._kernel = self.add_variable(
    #     _WEIGHTS_VARIABLE_NAME,
    #     shape=[input_depth + h_depth, self._num_splits * self._num_units],
    #     initializer=self._initializer,
    #     partitioner=maybe_partitioner)

    # attention in inputs
    self._kernel_attention = self.add_variable(
        "_kernel_attention",
        shape=[self._n_seg,2*self._num_units],
        initializer=self._initializer,#tf.keras.initializers.TruncatedNormal(mean=-2.0,stddev=0.25),#self._initializer,#init_ops.zeros_initializer,#
        partitioner=maybe_partitioner)
    self._bias_attention = self.add_variable(
        "_bias_attention",
        shape=[2* self._num_units],
        initializer=init_ops.zeros_initializer)


    self._kernel_context = self.add_variable(
        "traffic_context/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_var*self._n_seg],# * self._n_seg
        initializer=self._initializer,#tf.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.25),#tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
        partitioner=maybe_partitioner)
    self._bias_context = self.add_variable(
        "traffic_context/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_var*self._n_seg],# * self._n_seg
        initializer=init_ops.zeros_initializer)

    self._kernel_context2 = self.add_variable(
        "traffic_context2/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_var*self._n_seg,self._num_var*self._n_seg],# * self._n_seg
        initializer=self._initializer,#tf.keras.initializers.RandomUniform(minval=-1., maxval=2.0),#tf.keras.initializers.TruncatedNormal(mean=0.07,stddev=0.03),#tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),#
        partitioner=maybe_partitioner)
    self._bias_context2 = self.add_variable(
        "traffic_context2/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_var*self._n_seg],# * self._n_seg
        initializer=init_ops.zeros_initializer)

    self._kernel_outm = self.add_variable(
        "traffic_outm/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_units],
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

    # inputs_scaled = tf.truediv((inputs+1e-12),(self._max_values+1e-6))
    # m_prev_scaled = tf.truediv((m_prev+1e-12),(self._max_values+1e-6))
    # inputs = tf.Print(inputs,[inputs,tf.math.reduce_mean(inputs),tf.math.reduce_max(inputs),tf.math.reduce_min(inputs)],"inputs1",summarize=10,first_n=50)
    # stacked_inputs = tf.stack([inputs, m_prev], axis=2)
    # att_a = tf.stack([-sigmoid(inputs*10+10.0),sigmoid(inputs*10-10.0)],axis=2) #* att_c#tf.math.tanh
    un_inputs = tf.multiply(inputs,self._max_values+1e-6)
    att_a = sigmoid(-(un_inputs*1e15-1e9))#tf.stack([sigmoid((inputs*1e15-1e9)),sigmoid(-(inputs*1e15-1e9))],axis=2) #* att_c#tf.math.tanh
    inputs2 = inputs + m_prev*att_a#tf.reduce_sum(stacked_inputs * att_a,axis=2)
    # inputs = tf.Print(inputs,[inputs,tf.math.reduce_mean(inputs),tf.math.reduce_max(inputs),tf.math.reduce_min(inputs)],"inputs2",summarize=10,first_n=50)

    lstm_matrix = math_ops.matmul(inputs2, self._kernel)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # lstm_matrix = math_ops.matmul(
    #     array_ops.concat([inputs, m_prev], 1), self._kernel) #128+90 going in, 5*128 coming out [input_depth + h_depth, 5 * self._num_units]
        # self._kernel = self.add_variable(
            # _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + h_depth, 5 * self._num_units],
            # initializer=self._initializer,
            # partitioner=maybe_partitioner)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o  = array_ops.split(
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

    # # m = tf.layers.dropout(m,rate=0.5)
    # new_m = math_ops.matmul(m, self._kernel_outm)
    # new_m = tf.nn.relu(nn_ops.bias_add(new_m, self._bias_outm))
    # # new_m = tf.Print(new_m,[new_m,tf.math.reduce_mean(new_m),tf.math.reduce_max(new_m),tf.math.reduce_min(new_m)],"new_m",summarize=10,first_n=50)
    # #
    # # m = new_m

    ntf_matrix = math_ops.matmul(m, self._kernel_context)
    ntf_matrix = tf.nn.relu(nn_ops.bias_add(ntf_matrix, self._bias_context))

    # ntf_matrix = tf.layers.dropout(ntf_matrix,rate=0.5)
    # ntf_matrix = math_ops.matmul(ntf_matrix, self._kernel_context2)
    # ntf_matrix = tf.nn.relu(nn_ops.bias_add(ntf_matrix, self._bias_context2))

    ntf_matrix = tf.Print(ntf_matrix,[ntf_matrix,tf.math.reduce_mean(ntf_matrix),tf.math.reduce_max(ntf_matrix),tf.math.reduce_min(ntf_matrix)],"ntf_matrix",summarize=10,first_n=50)



    # boundry, traffic_variables = array_ops.split(value=ntf_matrix, num_or_size_splits=[4,self._num_var*self._n_seg], axis=1)#*self._n_seg
    # # boundry = tf.reshape(boundry,[-1,2,2])#32,2,2
    # # boundry = tf.multiply(boundry,[20000.,500.])#tf.multiply(boundry,[tf.constant(self._max_values[0]),tf.constant(self._max_values[1])])
    # # boundry = tf.Print(boundry,[boundry,tf.math.reduce_mean(boundry)],"boundry",summarize=10,first_n=50)
    # # contexts = ntf_matrix#array_ops.split(value=ntf_matrix, num_or_size_splits=self._n_seg, axis=1)

    traffic_variables = tf.reshape(ntf_matrix,[-1,self._n_seg,self._num_var])

    #attention
    # att_c = math_ops.matmul(traffic_variables[:,:,12], self._kernel_attention)
    # att_c = nn_ops.bias_add(att_c, self._bias_attention)
    # att_c = tf.reshape(att_c,[-1,self._num_units,2])
    # att_c = tf.reshape(att_c,[-1,self._num_units])
    # att_c = tf.nn.relu(att_c)
    # att_c = sigmoid(att_c)
    # att_c = tf.nn.tanh(att_c)
    # att_c = tf.stop_gradient(att_c)
    #att_c = tf.clip_by_value(att_c,0.0,1.0)
    #att_c= tf.Print(att_c,[att_c,tf.shape(att_c)],"att_c",summarize=10,first_n=10)

    # att_c = [1.,1.]*tf.nn.softmax(att_c,axis=-1) #-1 is actually default
    #
    # att_c = tf.Print(att_c,[att_c,tf.shape(att_c)],"att_c",summarize=10,first_n=10)
    # m_prev= tf.Print(m_prev,[m_prev,tf.shape(m_prev)],"m_prev",summarize=10,first_n=10)
    # inputs = tf.Print(inputs,[inputs,tf.shape(inputs)],"inputs",summarize=10,first_n=10)
    # # self._max_values = tf.Print(self._max_values,[self._max_values,tf.shape(self._max_values)],"_max_values",summarize=10,first_n=10)
    #
    # stacked_inputs = tf.stack([inputs, m_prev], axis=2)
    # stacked_inputs = tf.Print(stacked_inputs,[stacked_inputs,tf.shape(stacked_inputs)],"stacked_inputs",summarize=10,first_n=10)
    # unscaled_inputs = tf.reduce_sum(tf.multiply(stacked_inputs,att_c),axis=2) #m_prev
    # unscaled_inputs = m_prev + tf.multiply(att_c,inputs)#boundry[:,2:3]*inputs + (1.-boundry[:,2:3])*m_prev #tf.multiply(stacked_inputs,att_c)

    #residual add tanh

    # att_a = tf.stack([sigmoid(inputs*10-10.0),sigmoid(-inputs*10+10.0)],axis=2) #* att_c#tf.math.tanh
    # att_c = tf.Print(att_c,[att_c,tf.shape(att_c)],"att_c",summarize=10,first_n=10)
    #att_b = tf.math.tanh(inputs)
    # att_b = tf.reshape(traffic_variables[:,:,5:5+5],[-1,60])
    # att_b = sigmoid(-att_b-3.0)
    # att_b = tf.Print(att_b,[att_b,tf.shape(att_b)],"att_b",summarize=10,first_n=10)
    # norm_add = tf.multiply(att_c,m_prev_scaled)
    # norm_in = inputs_scaled + norm_add
    # norm_add_scaleup = tf.multiply((self._max_values+1e-6),norm_add)
    # unscaled_inputs = norm_in*self._max_values# + norm_add #+ tf.multiply(att_c,m_prev)
    # inputs = tf.stop_gradient(inputs)
    #m_prev = tf.stop_gradient(m_prev)
    #att_c = tf.stop_gradient(att_c)
    # unscaled_inputs = tf.reduce_sum(stacked_inputs * att_a,axis=2)
    # unscaled_inputs = inputs + tf.multiply(att_c,m_prev)
    # unscaled_inputs = inputs+ tf.reduce_mean(att_c)*0 + 0*m_prev#*(self._max_values+1e-6)
    # unscaled_inputs = tf.Print(unscaled_inputs,[unscaled_inputs,tf.math.reduce_max(unscaled_inputs)],"unscaled_inputs1",summarize=10,first_n=10)
    # unscaled_inputs = tf.reshape(unscaled_inputs,[-1,self._n_seg,5])#32,45,2
    unscaled_inputs = tf.multiply(inputs2,self._max_values+1e-6)
    unscaled_inputs = tf.reshape(unscaled_inputs,[-1,self._n_seg,5])
    unscaled_inputs = tf.Print(unscaled_inputs,[unscaled_inputs,tf.math.reduce_mean(unscaled_inputs),tf.math.reduce_max(unscaled_inputs),tf.math.reduce_min(unscaled_inputs)],"unscaled_inputs",summarize=10,first_n=50)


    flow_to_hr = tf.constant(120.0,name="flow_to_hr")#*tf.reduce_mean(traffic_variables[:,:,6]) #from (0,1) to.. 3000
    flow_scaling = tf.constant(20.0,name="flow_scaling") #high to scale 0,1 to
    density_scaling = tf.constant(3.0,name="density_scaling")#*tf.reduce_mean(traffic_variables[:,:,7])

    #flow_to_hr = tf.Print(flow_to_hr,[flow_to_hr,tf.math.reduce_mean(flow_to_hr),tf.math.reduce_max(flow_to_hr),tf.math.reduce_min(flow_scaling)],"flow_scaling",summarize=10,first_n=50)
    #density_scaling = tf.Print(density_scaling,[density_scaling,tf.math.reduce_mean(density_scaling),tf.math.reduce_max(density_scaling),tf.math.reduce_min(density_scaling)],"density_scaling",summarize=10,first_n=50)


    # v_f = tf.constant(200.,name="v_f") * tf.expand_dims(tf.reduce_mean(traffic_variables[:,:,0],axis=1),-1)
    # a = tf.constant(3.0,name="a")  *tf.expand_dims(tf.reduce_mean(traffic_variables[:,:,1],axis=1),-1)
    # p_cr = tf.constant(300.0,name="pcr") * tf.expand_dims(tf.reduce_mean(traffic_variables[:,:,2],axis=1),-1)
    v_f = tf.constant(120.,name="v_f") * traffic_variables[:,:,0]
    v_f =  tf.clip_by_value(v_f,90.0,120.0)
    a = tf.constant(1.4324,name="a")  * traffic_variables[:,:,1]
    a =  tf.clip_by_value(a,0.5,2.5)
    p_cr = tf.constant(33.5,name="pcr") * traffic_variables[:,:,2]
    p_cr =  tf.clip_by_value(p_cr,1.0,200.0)

    future_r_in = traffic_variables[:,:,3]*flow_scaling#*flow_scaling#tf.truediv(flow_scaling * traffic_variables[:,:,3],120.0)
    future_r_out = traffic_variables[:,:,4]*flow_scaling#*flow_scaling#tf.truediv(flow_scaling * traffic_variables[:,:,4],120.0)

    future_r_in = tf.Print(future_r_in,[future_r_in,tf.math.reduce_mean(future_r_in),tf.math.reduce_max(future_r_in),tf.math.reduce_min(future_r_in)],"future_r_in",summarize=10,first_n=50)
    future_r_out = tf.Print(future_r_out,[future_r_out,tf.math.reduce_mean(future_r_out),tf.math.reduce_max(future_r_out),tf.math.reduce_min(future_r_out)],"future_r_out",summarize=10,first_n=50)


    # g = tf.constant(10.0,name="g") * traffic_variables[:,:,5]

    lane_num = tf.constant(3.0,name="lane_num") #* traffic_variables[:,:,6]
    T = tf.truediv(tf.constant(10.0,name="T"),3600.0)# * traffic_variables[:,:,7]
    seg_len = tf.truediv(self._seg_lens,1000.0)#*tf.exp(traffic_variables[:,:,7])# tf.truediv(self._seg_lens,1000.0)
    # tau = tf.constant(30.,name="tau") * traffic_variables[:,:,8]
    # nu = tf.constant(100.,name="nu") * traffic_variables[:,:,9]
    # kappa = tf.constant(30.,name="kappa") * traffic_variables[:,:,10]
    # delta = tf.constant(3.0,name="delta") * traffic_variables[:,:,11]
    #Fixed variables
    # T = tf.constant(10.0,name="T") #* traffic_variables[:,:,7]
    # seg_len = self._seg_lens
    tau =  tf.truediv(tf.constant(20.,name="tau"),3600.0)#tf.constant(12.,name="tau")# * traffic_variables[:,:,8]
    nu = tf.constant(35.,name="nu") #* traffic_variables[:,:,9]
    kappa = tf.constant(13.,name="kappa") #* traffic_variables[:,:,10]
    delta = tf.constant(1.4,name="delta") #* traffic_variables[:,:,11]

    # lane_num = tf.Print(lane_num,[lane_num,tf.math.reduce_max(lane_num),tf.shape(lane_num)],"lane_num",summarize=10,first_n=10)
    # seg_len = tf.Print(seg_len,[seg_len,tf.math.reduce_max(seg_len),tf.shape(seg_len)],"seg_len",summarize=10,first_n=10)



    current_flows = tf.multiply(unscaled_inputs[:,:,0],flow_to_hr)
    current_velocities = unscaled_inputs[:,:,2]

    g = tf.constant(5.0,name="init_g")*tf.reshape(tf.reduce_mean(traffic_variables[:,:,5],1),[-1,1])#tf.expand_dims(tf.reduce_mean(traffic_variables[:,:,5],axis=-1),-1)#0.3#0.06#
    current_densities = tf.multiply(unscaled_inputs[:,:,1],g)#+0.0*tf.truediv(current_flows, current_velocities*lane_num + 1e-6) # tf.multiply(unscaled_inputs[:,:,1], g)#tf.truediv(current_flows, current_velocities*lane_num + 1e-6) #unscaled_inputs[:,:,1] * g#tf.truediv(current_flows, current_velocities*lane_num + 1e-6)
    # g = tf.truediv(current_densities,unscaled_inputs[:,:,2] + 1e-6)#* traffic_variables[:,:,5]
    g = tf.Print(g,[g,tf.math.reduce_mean(g)],"g",summarize=10,first_n=10)
    g =  tf.clip_by_value(g,0.02,100.0)


    r_in = tf.multiply(unscaled_inputs[:,:,3],flow_to_hr)
    r_out = tf.multiply(unscaled_inputs[:,:,4],flow_to_hr)

    # first_flow     = tf.multiply(boundry[:,0:1],flow_to_hr*flow_scaling)#flow_scaling) #current_flows[:,:1] #: variable
    # first_density  = tf.multiply(boundry[:,1:2],density_scaling)#,density_scaling) #current_densities[:,:1] #: variable
    # first_velocity = tf.multiply(boundry[:,3:4],100.0)#tf.truediv(first_flow,first_density*lane_num)#tf.clip_by_value(tf.multiply(boundry[:,3:4],120.0),1.0,120.0)#tf.truediv(first_flow,first_density*lane_num)#[:,0:1]) #lane_num is [batch_size,12]
    # last_density   = tf.multiply(boundry[:,2:3],density_scaling) #current_densities[:,-1:] #: variable

    first_flow     = tf.multiply(tf.reduce_mean(traffic_variables[:,:,12],1),flow_to_hr*flow_scaling)
    first_density  = tf.multiply(tf.reduce_mean(traffic_variables[:,:,13],1),density_scaling)
    first_velocity = tf.multiply(tf.reduce_mean(traffic_variables[:,:,14],1),100.0)
    last_density   = tf.multiply(tf.reduce_mean(traffic_variables[:,:,15],1),density_scaling)

    first_flow     = tf.reshape(first_flow,[-1,1])
    first_density  = tf.reshape(first_density,[-1,1])
    first_velocity = tf.reshape(first_velocity,[-1,1])
    last_density   = tf.reshape(last_density,[-1,1])




    first_flow = tf.Print(first_flow,[first_flow,tf.math.reduce_mean(first_flow)],"first_flow",summarize=10,first_n=10)
    first_density = tf.Print(first_density,[first_density,tf.math.reduce_mean(first_density)],"first_density",summarize=10,first_n=10)
    first_velocity = tf.Print(first_velocity,[first_velocity,tf.math.reduce_mean(first_velocity)],"first_velocity",summarize=10,first_n=10)
    last_density = tf.Print(last_density,[last_density,tf.math.reduce_mean(last_density)],"last_density",summarize=10,first_n=10)
    # future_r_in = tf.Print(future_r_in,[future_r_in,tf.math.reduce_mean(future_r_in)],"future_r_in",summarize=10,first_n=10)
    # future_r_out = tf.Print(future_r_out,[future_r_out,tf.math.reduce_mean(future_r_out)],"future_r_out",summarize=10,first_n=10)


    prev_flows      = tf.concat([first_flow,current_flows[:,:-1]],1)
    prev_densities  = tf.concat([first_density,current_densities[:,:-1]],1)
    prev_velocities = tf.concat([first_velocity,current_velocities[:,:-1]],1)
    next_densities  = tf.concat([current_densities[:,1:],last_density],1)


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
        future_rho =   tf.clip_by_value(future_rho,0.01,1000.0)
        future_rho = tf.Print(future_rho,[future_rho,tf.math.reduce_max(future_rho),tf.shape(future_rho)],"future_rho",summarize=10,first_n=10)#[32 45]

    with tf.name_scope("future_velocity"):
        stat_speed =  tf.multiply( v_f, tf.exp( (tf.multiply(tf.truediv(-1.0,a),tf.math.pow(tf.truediv(current_densities,p_cr),a)))))
        stat_speed =  tf.clip_by_value(stat_speed,5.0,120.0)
        stat_speed = tf.Print(stat_speed,[stat_speed,tf.math.reduce_max(stat_speed),tf.shape(stat_speed)],"stat_speed",summarize=10,first_n=10)

        future_vel = current_velocities + ( (T/tau) * (stat_speed - current_velocities) )\
                        + ( (current_velocities*T/seg_len) * (prev_velocities - current_velocities ) )\
                        - ( (nu*T/(tau*seg_len)) * ( (next_densities - current_densities) / (current_densities + kappa) )  )\
                        - ( (delta*T/(seg_len*lane_num)) * ( (r_in * current_velocities) / (current_densities+kappa) ) )
        future_vel = tf.Print(future_vel,[future_vel,tf.math.reduce_max(future_vel),tf.shape(future_vel)],"future_vel",summarize=10,first_n=10)#[32,45]
        future_vel =   tf.clip_by_value(future_vel,40.0,111.)


    future_flows = tf.multiply(future_rho,future_vel*lane_num)#tf.divide(tf.multiply(future_rho,future_vel),tf.constant(4.0))

    future_volumes = tf.truediv(future_flows,flow_to_hr)
    future_occupancies = tf.truediv(future_rho,g)#tf.truediv(future_rho,g+1e-6)

    future_volumes = tf.Print(future_volumes,[future_volumes,tf.math.reduce_max(future_volumes),tf.shape(future_volumes)],"future_volumes",summarize=10,first_n=10)
    future_occupancies = tf.Print(future_occupancies,[future_occupancies,tf.math.reduce_max(future_occupancies),tf.shape(future_occupancies)],"future_occupancies",summarize=10,first_n=10)

    future_states = tf.stack([future_volumes,future_occupancies,future_vel,future_r_in,future_r_out],axis=2)

    future_states = tf.reshape(future_states,[-1,5*self._n_seg])
    new_m = tf.truediv(tf.clip_by_value(future_states,0.0,100.0), (self._max_values+1e-6))

    # m = tf.truediv(tf.clip_by_value(future_states,0.1,120.0) , (self._max_values+1e-6))
    # new_m = tf.Print(new_m,[new_m,tf.math.reduce_mean(new_m),tf.math.reduce_max(new_m),tf.math.reduce_min(new_m)],"new_m",summarize=10,first_n=50)
    m = tf.clip_by_value(new_m,0.0,10.0)
    m = new_m

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
