from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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
               activation=None, reuse=None, name=None, n_seg=45, dtype=None, **kwargs):
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
    self._n_seg = n_seg
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
    max_vals = [46,24,136,23.8333,136,23.8333,175,19.2778,97,27.5,63,13.6667,454,24.5,899,37.25,850,67,1262,57.8571,2064,56.9231,87,55.6667,804,66,1362,67.3333,472,83.6667,560,68.3333,563,74.1667,563,74.1667,952,70.8333,437,60.9583,1275,61.2222,884,78.3333,394,80.5,394,80.5,189,84,325,49.4167,616,44.5,651,60,718,59,1546,54.1667,1546,60.1667,796,77.25,1356,67.3333,772,69.1667,162,80.6667,162,80.6667,485,55,307,56.525,293,51.75,271,56.65,308,56.65,299,59.2857,250,64.8,126,94,126,94]
    self._max_values=tf.convert_to_tensor(max_vals)


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
        shape=[input_depth + h_depth, 5 * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)

    self._kernel_context = self.add_variable(
        "traffic_context/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, 16 * self._n_seg])
    self._bias_context = self.add_variable(
        "traffic_context/%s" % _BIAS_VARIABLE_NAME,
        shape=[16 * self._n_seg])


    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[5 * self._num_units],
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

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev], 1), self._kernel) #128+90 going in, 5*128 coming out [input_depth + h_depth, 5 * self._num_units]
        # self._kernel = self.add_variable(
            # _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + h_depth, 5 * self._num_units],
            # initializer=self._initializer,
            # partitioner=maybe_partitioner)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o, t  = array_ops.split(
        value=lstm_matrix, num_or_size_splits=5, axis=1)
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

    # inputs should be around
    unscaled_inputs = self._max_values*inputs
    # unscaled_inputs = tf.reshape(unscaled_inputs,[2,-1])
    unscaled_inputs = tf.Print(unscaled_inputs,[tf.shape(unscaled_inputs)],"unscaled_inputs")
    each_seg = array_ops.split(value=unscaled_inputs, num_or_size_splits=self._n_seg, axis=1)


    # c = tf.layers.dense(t,self._n_seg*5)
    ntf_matrix = math_ops.matmul(m, self._kernel_context)
    ntf_matrix = nn_ops.bias_add(ntf_matrix, self._bias_context)

    contexts = array_ops.split(value=ntf_matrix, num_or_size_splits=self._n_seg, axis=1)

    def next_density(in_vars):
        with tf.name_scope("next_density"):
            x0,x1,x2,c1 = in_vars#array_ops.split(value=in_vars, num_or_size_splits=4, axis=1)#in_vars
            # return tf.clip_by_value(tf.div(in_vars[0],in_vars[1]+1e-6),0.0,120.0)
            current_density = x1[1]
            T = c1[4]#tf.constant(1.0,name="T")*
            seg_len = c1[5]#tf.constant(1000.0,name="seg_len")*
            lane_num = c1[6]#tf.constant(3.0,name="lane_num")*
            flow0 = x0[0]
            flow1 = x1[0]
            r_in = c1[0]
            r_out = c1[1]
        rho =  current_density + tf.divide(T,seg_len*lane_num) * (flow0 - flow1 + r_in - r_out)
        return rho*c1[13]

    def velocity_fn(x):
        return tf.clip_by_value(tf.div(x[0],x[1]+1e-6),0.0,150.0)

    def next_velocity(in_vars):
        with tf.name_scope("next_velocity"):
            x0,x1,x2,c1 = in_vars
            current_velocity = velocity_fn(x1)
            current_density = x1[1]
            seg_len = c1[5]#*tf.constant(1000.0,name="seg_len_v")*
            lane_num = c1[6]#*tf.constant(3.0,name="lane_num_v")*
            T = c1[4]#tf.constant(1.,name="T_v")*
            tau = c1[7] #tf.constant(120.,name="tau")*
            nu = c1[8] #tf.constant(35.,name="nu")*
            kappa = c1[9] #tf.constant(13.,name="kappa")*
            delta = c1[10] #tf.constant(1.4,name="delta")*
            v_f = c1[11] #tf.constant(120.,name="v_f")*
            a = tf.clip_by_value(c1[2],0.5,1.9)
            p_cr = c1[3]
            r_in = c1[0]

            stat_speed =  v_f * tf.exp( (-1/a) * (current_density*p_cr) **a )
            # stat_speed = tf.Print(stat_speed,[stat_speed],"stat_speed")

            next_vel = current_velocity + ( (T/tau) * (stat_speed - current_velocity) )\
                            + ( (current_velocity*T/seg_len) * (velocity_fn(x0) - current_velocity ) )\
                            - ( (nu*T/(tau*seg_len)) * ( (x2[1] - current_density) / (current_density + kappa) )  )\
                            - ( (delta*T/(seg_len*lane_num)) * ( (r_in * current_velocity) / (current_density+kappa) ) )
        return tf.clip_by_value(next_vel,0.0,150.0)*c1[12]



    #tf.layers.dense(x,1)
    next_v = lambda x: next_velocity(x)
    next_rho = lambda x: next_density(x)
    # vs = []
    next_states = []

    in_vars = (each_seg[0],each_seg[0],each_seg[1],contexts[0])
    velocity = tf.map_fn(next_v,in_vars, dtype=tf.float32)
    density = tf.map_fn(next_rho,in_vars, dtype=tf.float32)
    next_states.append((density*velocity))
    next_states.append(density)

    for g in range(1,len(each_seg)-1):
        # seg = each_seg[g]
        # in_vars = array_ops.concat([each_seg[g-1],each_seg[g],each_seg[g+1],contexts[g]],1)
        in_vars = (each_seg[g-1],each_seg[g],each_seg[g+1],contexts[g])
        # vs.append(tf.map_fn(next_rho,in_vars, dtype=tf.float32))

        velocity = tf.map_fn(next_v,in_vars, dtype=tf.float32)
        density = tf.map_fn(next_rho,in_vars, dtype=tf.float32)
        # velocity = tf.Print(velocity,[velocity],"velocity")
        # density = tf.Print(density,[density],"density")
        next_states.append((density*velocity))
        next_states.append(density)

    last_seg = len(each_seg)-1
    in_vars = (each_seg[last_seg-1],each_seg[last_seg],each_seg[last_seg],contexts[last_seg])
    velocity = tf.map_fn(next_v,in_vars, dtype=tf.float32)
    density = tf.map_fn(next_rho,in_vars, dtype=tf.float32)
    next_states.append((density*velocity))
    next_states.append(density)

    next_states = tf.stack(next_states,axis=1)
    # next_states = tf.stack(vs,axis=1)
    # vs.set_shape([None,45])
    # next_states = tf.clip_by_value(next_states/self._max_values,0.0,2.0)
    # next_states = sigmoid(next_states)
    next_states = next_states / self._max_values
    # vs = sigmoid(vs)
        # gradients, variables = zip(*optimizer.compute_gradients(loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # 25%| 442/1799 [2:37:54<8:03:51, 21.39s/it, loss=2774.847]
    m = tf.layers.dense(next_states,self._num_units,activation='tanh')
    # m = tf.layers.dense(array_ops.concat([next_states, m],1),self._num_units)


    if self._num_proj is not None:
      m = math_ops.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

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
