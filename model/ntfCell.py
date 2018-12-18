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
               activation=None, reuse=None, name=None, n_seg=5,num_var = 11, dtype=None, **kwargs):
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
    self._num_var = num_var
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
    # max_vals = [46,24,136,23.8333,136,23.8333,175,19.2778,97,27.5,63,13.6667,454,24.5,899,37.25,850,67,1262,57.8571,2064,56.9231,87,55.6667,804,66,1362,67.3333,472,83.6667,560,68.3333,563,74.1667,563,74.1667,952,70.8333,437,60.9583,1275,61.2222,884,78.3333,394,80.5,394,80.5,189,84,325,49.4167,616,44.5,651,60,718,59,1546,54.1667,1546,60.1667,796,77.25,1356,67.3333,772,69.1667,162,80.6667,162,80.6667,485,55,307,56.525,293,51.75,271,56.65,308,56.65,299,59.2857,250,64.8,126,94,126,94]
    # max_vals = [1839.9999948889,23.9999996476,5439.9999848889,45.3333332074,5439.9999848889,45.3333332074,6999.9999805556,58.3333331713,3879.9999892222,32.3333332435,2519.999993,20.9999999417,18159.9999495556,151.333332913,35959.9999001111,299.6666658343,33999.9999055556,283.3333325463,50479.9998597778,420.6666654981,82559.9997706667,687.9999980889,3479.9999903333,55.6666974641,32159.9999106667,267.9999992556,54479.9998486667,453.9999987389,18879.9999475556,157.3333328963,22399.9999377778,\
    # 186.6666661481,22519.9999374444,187.6666661454,22519.9999374444,187.6666661454,38079.9998942222,317.3333324519,17479.9999514444,145.666666262,50999.9998583333,424.9999988194,35359.9999017778,294.6666658481,15759.9999562222,131.3333329685,15759.9999562222,131.3333329685,7559.999979,83.9999905333,8839.9999754445,73.666666462,24639.9999315556,205.333332763,26039.9999276667,216.9999993972,28719.9999202222,239.3333326685,61839.9998282222,515.3333319019,61839.9998282222,515.3333319019,\
    # 31839.9999115556,265.3333325963,54239.9998493333,451.9999987444,30879.9999142222,257.3333326185,6199.9999827778,80.6666943082,6199.9999827778,80.6666943082,19399.9999461111,161.6666662176,12279.9999658889,102.3333330491,11719.9999674444,97.6666663954,10839.9999698889,90.3333330824,12319.9999657778,102.6666663815,11959.9999667778,99.6666663898,9999.9999722222,83.3333331019,5039.999986,92.4999646823,5039.999986,92.4999646823]
    max_vals = [22520.0,188.0,38080.0,317.5,17480.0,145.5,51000.0,425.0,35360.0,295.0]
    #[5520.0046,24.0024,16320.0136,23.83568333,16320.0136,23.83568333,21000.0175,19.27972778,11640.0097,27.50275,7560.0063,13.0013,54480.0454,24.50245,107880.0899,37.253725,102000.085,67.0067,151440.1262,57.86288571,247680.2064,56.92879231,10440.0087,55.67226667,96480.0804,66.0066,163440.1362,67.34003333,56640.0472,83.67506667,67200.056,64.83978333,67560.0563,74.17411667,67560.0563,74.17411667,114240.0952,70.84038333,52440.0437,57.755775,153000.1275,61.22832222,106080.0884,78.34113333,\
    # 47280.0394,80.50805,47280.0394,80.50805,22680.0189,84.0084,26520.0221,40.58735833,73920.0616,36.0036,78120.0651,60.006,86160.0718,59.0059,185520.1546,54.17211667,185520.1546,60.17271667,95520.0796,77.257725,162720.1356,67.34003333,92640.0772,69.17361667,18600.0155,80.67476667,18600.0155,80.67476667,58200.0485,55.0055,36840.0307,56.5306525,35160.0293,51.755175,32520.0271,56.655665,36960.0308,56.655665,35880.0299,59.29162857,30000.025,64.80648,15120.0126,92.50925,15120.0126,92.50925]
    self._max_values=tf.convert_to_tensor(max_vals)

    all_seg_lens = [900.0,600.0,900.0,700.0,1800.0]
    self._seg_lens =tf.convert_to_tensor(all_seg_lens)
    # max_vals = [ 4462.11559904,    53.7349221 ,  4423.03294541,    73.53982445,
    #     4327.15055338,    89.10455954,  4251.69290532,    95.81514881,\
    #     4234.36138235,    92.84096884,  4222.62811687,    88.3101285 ,\
    #     4210.2520675 ,    85.26524715,  4199.48273431,    83.25366832,\
    #     4190.6376384 ,    81.86985297,  4183.43948374,    80.87742503,\
    #     4177.52964524,    80.1290634 ,  4172.54858902,    79.53841533,\
    #     4168.19515141,    79.04591297,  4164.22763872,    78.61642646,\
    #     4160.505622  ,    78.22611144,  4156.94038752,    77.86309195,\
    #     4155.06841303,    77.67906958,  4153.9971503 ,    77.5714677 ,\
    #     4152.54931464,    77.42943482,  4150.77507396,    77.25851049,\
    #     4148.7331974 ,    77.06446511,  4146.47278869,    76.85389496,\
    #     4144.05415523,    76.63130488,  4141.51298844,    76.40176305,\
    #     4138.88609777,    76.16900624,  4136.20437434,    75.9356977 ,\
    #     4133.49762237,    75.70367378,  4130.77175486,    75.47414447,\
    #     4128.03608594,    75.24784166,  4125.31657027,    75.02634983,\
    #     4122.60359952,    74.80922699,  4119.90433693,    74.59603951,\
    #     4117.22659536,    74.38842582,  4114.56360709,    74.18416312,\
    #     4111.92018099,    73.98533335,  4109.30049486,    73.78988858,\
    #     4106.69348445,    73.59862985,  4104.10875128,    73.41160488,\
    #     4101.55028946,    73.22803105,  4099.00974622,    73.04797625,\
    #     4096.48912851,    72.87159771,  4093.98995057,    72.699158  ,\
    #     4091.51329988,    72.53005523,  4089.05995335,    72.36400068,\
    #     4086.65098963,    72.10656196]
    # [ 1434.74750643,    18.30139494,  1429.74008344,    17.48155306,\
    #         1426.702155  ,    17.99523301,  1424.37520692,    18.08119266,\
    #         1422.62415457,    18.07461953,  1421.35611114,    18.05625932,\
    #         1420.45478819,    18.04031743,  1419.90337358,    18.03026502,\
    #         1419.52384034,    18.02317216,  1419.20911214,    18.01715104,\
    #         1418.95221455,    18.01225176,  1418.70013052,    18.00769527,\
    #         1418.43693812,    18.00269478,  1418.14779313,    17.99711155,\
    #         1417.81579209,    17.99103296,  1417.44840569,    17.98407964,\
    #         1417.06875011,    17.97687155,  1416.64375747,    17.96918448,\
    #         1416.22762397,    17.96095446,  1415.7796153 ,    17.95286074,\
    #         1415.32413225,    17.9440461 ,  1414.86735724,    17.93563274,\
    #         1414.38253656,    17.92676427,  1413.92731718,    17.91788703,\
    #         1413.4421255 ,    17.90913208,  1412.97392148,    17.89990094,\
    #         1412.50099183,    17.89137473,  1412.01763006,    17.88237556,\
    #         1411.56109772,    17.8736535 ,  1411.08129417,    17.86500379,\
    #         1410.62835788,    17.85608305,  1410.16940487,    17.84781971,\
    #         1409.70701819,    17.8391717 ,  1409.26995687,    17.83088746,\
    #         1408.81330982,    17.82266766,  1408.3853254 ,    17.81425268,\
    #         1407.95186431,    17.8064622 ,  1407.51721571,    17.79833338,\
    #         1407.10663076,    17.79058055,  1406.67835256,    17.78288625,\
    #         1406.27857516,    17.77504053,  1405.87307208,    17.76776747,\
    #         1405.46838831,    17.76017997,  1405.08484042,    17.7529909 ,\
    #         1404.7018785 ,    17.75292314]
    # self._max_values=tf.convert_to_tensor(max_vals)


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

    self._kernel_context = self.add_variable(
        "traffic_context/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, 4 + self._num_var],# * self._n_seg
        initializer=self._initializer,#tf.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.25),#tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
        partitioner=maybe_partitioner)
    self._bias_context = self.add_variable(
        "traffic_context/%s" % _BIAS_VARIABLE_NAME,
        shape=[4 + self._num_var],# * self._n_seg
        initializer=init_ops.zeros_initializer)

    self._kernel_outm = self.add_variable(
        "traffic_outm/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[2*self._n_seg, self._num_units],
        initializer=tf.keras.initializers.Identity(gain=1.0),#self._initializer,#tf.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.25),#tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
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
        shape=[4 * self._num_units],
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

    inputs = inputs/self._max_values
    m_prev = m_prev/self._max_values
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev], 1), self._kernel) #128+90 going in, 5*128 coming out [input_depth + h_depth, 5 * self._num_units]
        # self._kernel = self.add_variable(
            # _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + h_depth, 5 * self._num_units],
            # initializer=self._initializer,
            # partitioner=maybe_partitioner)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o  = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
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

    # m = tf.Print(m,[m,tf.shape(m)],"m")
    # inputs should be around
    unscaled_inputs = inputs*self._max_values#*

    # vols = unscaled_inputs[:,0::2]
    # occs = unscaled_inputs[:,1::2]
    # spds = tf.clip_by_value(vols/(occs*tf.constant(3.0)+1e-6),0.,120.0)
    # dsty = vols/(tf.constant(3.0)*spds+1e-6)
    #
    # occs = dsty
    # vols = dsty*spds
    #
    # unscaled_inputs = tf.stack([vols,occs],axis=2)



    # unscaled_inputs = tf.Print(unscaled_inputs,[unscaled_inputs,tf.shape(unscaled_inputs)],"unscaled_inputs",summarize=10,first_n=10)
    # feature_mask = np.full((256,90), False)
    # feature_mask[:,::2*5] = True
    # feature_mask[:,1::2*5] = True
    # unscaled_inputs = tf.boolean_mask(unscaled_inputs, feature_mask)

    unscaled_inputs = tf.reshape(unscaled_inputs,[-1,self._n_seg,2])#32,45,2
    unscaled_inputs = tf.multiply(unscaled_inputs,[120.,30.])

    unscaled_inputs = tf.Print(unscaled_inputs,[unscaled_inputs,tf.shape(unscaled_inputs)],"unscaled_inputs2",summarize=10,first_n=10)
    # unscaled_inputs = tf.Print(unscaled_inputs,[tf.shape(unscaled_inputs)],"unscaled_inputs")
    # vols, occs = array_ops.split(value=unscaled_inputs, num_or_size_splits=2,axis=2)#self._n_seg, axis=1)
    # vols = tf.squeeze(vols)
    # occs = tf.squeeze(occs)
    #m = tf.Print(m,[m,tf.shape(m)],"m")

    ntf_matrix = math_ops.matmul(m, self._kernel_context)
    ntf_matrix = sigmoid(nn_ops.bias_add(ntf_matrix, self._bias_context))#)

    boundry,ntf_matrix = array_ops.split(value=ntf_matrix, num_or_size_splits=[4,self._num_var], axis=1)#*self._n_seg
    # boundry = tf.Print(boundry,[boundry,tf.shape(boundry)],"boundry")#[32 4]
    boundry = tf.reshape(boundry,[-1,2,2])#32,2,2
    boundry = tf.multiply(boundry,[51000.,500.])#tf.multiply(boundry,[tf.constant(self._max_values[0]),tf.constant(self._max_values[1])])
    # boundry = tf.Print(boundry,[boundry,tf.math.reduce_mean(boundry)],"boundry",summarize=10,first_n=50)
    # boundry = tf.Print(boundry,[boundry,tf.shape(boundry)],"boundry",summarize=12,first_n=50)
    contexts = ntf_matrix#array_ops.split(value=ntf_matrix, num_or_size_splits=self._n_seg, axis=1)

    eq_vars = tf.reshape(ntf_matrix,[-1,self._num_var,1])#+tf.constant(1.0)#*tf.constant(10.0,name="eq_vars")self._n_seg,
    # eq_vars = tf.Print(eq_vars,[eq_vars,tf.math.reduce_max(eq_vars)],"eq_vars",summarize=10,first_n=10)
    #eq_vars = tf.Print(eq_vars,[eq_vars,tf.shape(eq_vars)],"eq_vars")

    prev_segs = tf.concat([unscaled_inputs[:,0:1,:], unscaled_inputs[:,:-1,:]],1)#prev_segs = tf.concat([boundry[:,0:1,:], unscaled_inputs[:,:-1,:]],1)
    # prev_segs = tf.concat([boundry[:,0:1,:], unscaled_inputs[:,:-1,:]],1)
    #prev_segs = tf.Print(prev_segs,[prev_segs,tf.shape(prev_segs)],"prev_segs")
    # prev_occs = tf.concat([boundry[:,1:2], occs[:,:-1]],1)
    next_segs = tf.concat([unscaled_inputs[:,1:,:], unscaled_inputs[:,-1:,:]],1)#tf.concat([unscaled_inputs[:,1:,:], boundry[:,1:2,:]],1)
    # next_segs = tf.concat([unscaled_inputs[:,1:,:], boundry[:,1:2,:]],1)
    # next_occs = tf.concat([occs[:,1:], boundry[:,3:4]],1)

    """unscaled_inputs is the current_seg volume and density [32,45,2]
        prev_segs is previous timestep volume and density [32,45,2]
        next_segs is next timestep volume and density [32,45,2]
        eq_vars are the variables in equation (up to 16) [32,45,16]
    """
    T = tf.constant(10.0,name="T")#eq_vars[:,:,4]*tf.constant(60.0,name="T")#tf.constant(1.0,name="T")*
    seg_len = self._seg_lens*(tf.constant(1.0,name="sl")+eq_vars[:,5,0:1])#*tf.constant(6000.0,name="sl")#+tf.constant(1500.0,name="seg_len")#*

    current_volume = unscaled_inputs[:,:,0]#*tf.constant(4.0,name="q")
    current_density = unscaled_inputs[:,:,1]#*(tf.constant(0.5)+eq_vars[:,:,12])#*tf.constant(2.0)#+tf.constant(2.)*tf.constant(3.0)

    seg_len = tf.Print(seg_len,[seg_len,tf.shape(seg_len)],"seg_len",summarize=10,first_n=10)
    lane_num = tf.constant(10.0,name="lane_num")*eq_vars[:,6,0:1]#*tf.constant(6.0,name="lane_num")#*
    # flow0 = prev_segs[:,:,0]
    # flow1 = unscaled_inputs[:,:,0]
    r_in = tf.constant(30000.0,name="r_in")*eq_vars[:,7,0:1]#*tf.constant(1000.0,name="r_in")
    r_out = tf.constant(30000.0,name="r_out")*eq_vars[:,8,0:1]#*tf.constant(1000.0,name="r_out")

    prev_volume = prev_segs[:,:,0]#*tf.constant(4.0,name="q")
    prev_density = prev_segs[:,:,1]#*(tf.constant(0.5)+eq_vars[:,:,12])#*tf.constant(2.0)#tf.constant(2.0)#+tf.constant(10.)+*
    next_density = next_segs[:,:,1]#*(tf.constant(0.5)+eq_vars[:,:,12])#*tf.constant(2.0)
    current_velocity = tf.div(current_volume,current_density*lane_num+1e-6)#tf.clip_by_value(tf.div(current_volume,current_density*lane_num+1e-6),0.0,150.0)
    prev_velocity = tf.div(prev_volume,prev_density*lane_num+1e-6)#tf.clip_by_value(tf.div(prev_volume,prev_density*lane_num+1e-6),0.0,150.0)
    tau = tf.constant(12.,name="tau")#eq_vars[:,:,7]*tf.constant(100.,name="tau")#+tf.constant(1.0)#*20
    nu = tf.constant(35.,name="nu")#eq_vars[:,:,8]*tf.constant(100.,name="nu")#+tf.constant(1.0)# + tf.constant(35.,name="nu")#*
    kappa = tf.constant(13.,name="kappa")#eq_vars[:,:,9]*tf.constant(100.,name="kappa")#+tf.constant(1.0)# + tf.constant(13.,name="kappa")#*
    delta = tf.constant(1.4,name="delta")#eq_vars[:,:,10]*tf.constant(10.,name="delta")#+tf.constant(1.0)# + tf.constant(1.4,name="delta")#*
    v_f = tf.constant(20.,name="v_min")+eq_vars[:,0,0:1]*tf.constant(200.,name="v_f")#+tf.constant(1.0)# + tf.constant(120.,name="v_f")#eq_vars[:,0,0:1]
    v_f = tf.Print(v_f,[v_f,tf.math.reduce_mean(v_f)],"v_f",summarize=10,first_n=10)
    a = tf.constant(3.0,name="a")*eq_vars[:,2,0:1]#tf.clip_by_value(eq_vars[:,:,2]*tf.constant(2.0,name="a"),0.5,1.6)#tf.clip_by_value(eq_vars[:,:,2]+ tf.constant(1.,name="a"),0.5,1.5)#
    a = tf.Print(a,[a,tf.math.reduce_mean(a)],"a",summarize=10,first_n=10)
    p_cr = tf.constant(1000.0,name="pcr")*eq_vars[:,1,0:1]#*#+tf.constant(1.0)# + tf.constant(33.5,name="p_cr")#
    p_cr = tf.Print(p_cr,[p_cr,tf.math.reduce_mean(p_cr)],"p_cr",summarize=10,first_n=10)

    with tf.name_scope("next_density"):
        next_rho =  current_density + tf.multiply(tf.div(T,tf.multiply(seg_len,lane_num)),(prev_volume - current_volume + r_in - r_out))
        # next_rho = tf.Print(next_rho,[next_rho,tf.math.reduce_max(next_rho)],"ne`xt_rho",summarize=10,first_n=10)#[32 45]

    with tf.name_scope("next_velocity"):
        stat_speed =  tf.multiply( v_f, tf.exp( (tf.multiply(tf.div(-1.0,a),tf.math.pow(tf.divide(current_density,p_cr),a)))))
        # stat_speed = tf.Print(stat_speed,[stat_speed,tf.math.reduce_max(stat_speed)],"stat_speed",summarize=10,first_n=10)

        next_vel = current_velocity + ( (T/tau) * (stat_speed - current_velocity) )\
                        + ( (current_velocity*T/seg_len) * (prev_velocity - current_velocity ) )\
                        - ( (nu*T/(tau*seg_len)) * ( (next_density - current_density) / (current_density + kappa) )  )\
                        - ( (delta*T/(seg_len*lane_num)) * ( (r_in * current_velocity) / (current_density+kappa) ) )
        # next_vel = tf.Print(next_vel,[next_vel,tf.math.reduce_max(next_vel)],"next_vel",summarize=10,first_n=10)#[32,45]
        # next_vel = tf.clip_by_value(next_vel,0.0,150.0)


    next_flows = tf.multiply(next_rho,next_vel)#tf.divide(tf.multiply(next_rho,next_vel),tf.constant(4.0))

    next_flows = tf.divide(next_flows,120.0)
    next_rho = tf.divide(next_flows,30.0)#(Occs/(12.0/1000.0)) / lane_num

    # next_rho = tf.div(next_rho,(tf.constant(0.5)+eq_vars[:,:,12]))#*tf.constant(2.0))#*tf.constant(10.0))
    next_states = tf.stack([next_flows,next_rho],axis=2)
    # next_states = tf.Print(next_states,[next_states,tf.shape(next_states)],"next_states")
    next_states = tf.reshape(next_states,[-1,2*self._n_seg])
    # next_states = tf.Print(next_states,[next_states,tf.shape(next_states)],"next_states")
    # next_states = tf.clip_by_value(next_states,0.9,87000.0)

    # vs = sigmoid(vs)
        # gradients, variables = zip(*optimizer.compute_gradients(loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # 25%| 442/1799 [2:37:54<8:03:51, 21.39s/it, loss=2774.847]
    # m = tf.layers.dense(next_states,self._num_units,activation='tanh')
    # m = tf.layers.dense(array_ops.concat([next_states, m],1),self._num_units)
    # next_states = tf.Print(next_states,[next_states,tf.math.reduce_max(next_states)],"next_states",summarize=10,first_n=20)
    # next_states = next_states / (self._max_values)
    # next_states = tf.Print(next_states,[next_states,tf.math.reduce_max(next_states)],"next_states2",summarize=12,first_n=20)

    m = next_states #/ (self._max_values)
    # m = math_ops.matmul(next_states, self._kernel_outm)
    # m = self._activation(nn_ops.bias_add(m, self._bias_outm))
    # # m = tf.Print(m,[m,tf.shape(m)],"m",summarize=10,first_n=20)
    #
    # if self._num_proj is not None:
    #   m = math_ops.matmul(m, self._proj_kernel)
    #
    #   if self._proj_clip is not None:
    #     # pylint: disable=invalid-unary-operand-type
    #     m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
    #     # pylint: enable=invalid-unary-operand-type

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
        shape=[4 * self._num_units],
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
