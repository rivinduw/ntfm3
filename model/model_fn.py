"""Define the model."""
#model_fn.py stores the model architecture

import tensorflow as tf
import numpy as np

from model.ntfCell import ntfCell
from model.ntfCell import LSTMCell2
from tensorflow.python.framework import dtypes

def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    input_batch = inputs['input_batch']

    if params.model_version == 'lstm':
        lstm_cell = ntfCell(params.num_cols,num_var = 13,max_vals = params.max_vals, all_seg_lens = params.seg_lens)#,use_peepholes=True,cell_clip=3.0)
        # lstm_cell = LSTMCell2(params.lstm_num_units)#,use_peepholes=True,cell_clip=3.0)

        init_state = lstm_cell.zero_state(params.batch_size, dtype=tf.float32)
        init_state = tf.identity(init_state, 'init_state') #Actually it works without this line. But it can be useful
        _lstm_state_ = tf.contrib.rnn.LSTMStateTuple(init_state[0, :, :], init_state[1,:,:]+ params.mean_vals)

        # initial_state[1] = initial_state[1] + params.max_vals
        # initial_state = [(tf.add(state[0],tf.ones_like(state[0])*params.max_vals), state[1]) for state in initial_state]

        rnn_outputs, rnn_states  = tf.nn.dynamic_rnn(lstm_cell, input_batch, dtype=tf.float32,initial_state=_lstm_state_)

        # # Compute logits from the output of the LSTM
        # logits = tf.layers.dense(output, params.number_of_tags)
        # project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
        # an extra layer here.
        # final_projection = lambda x: tf.layers.dense(x,1)# params.rnn_output_size)
        # apply projection to every timestep.
        # predicted_outputs = tf.map_fn(final_projection, rnn_outputs)
        # predicted_outputs =  tf.layers.dense(rnn_outputs, params.rnn_output_size,activation='linear')
        predicted_outputs =  rnn_outputs#tf.layers.dense(rnn_outputs, params.rnn_output_size)

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return predicted_outputs


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['label_batch']
    # sentence_lengths = inputs['sentence_lengths']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        predicted_outputs = build_model(mode, inputs, params)
        # predictions = tf.argmax(logits, -1)




    # Define loss and accuracy (we need to apply a mask to account for padding)
    # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # print(labels.shape)
    # feature_mask = np.full((params.batch_size,params.window_size,params.num_cols), False)
    # feature_mask[:,::18,::5]=True#20*2:25*2:2] = True
    # feature_mask[:,::18,1::5]=True
    # losses = tf.boolean_mask(losses, feature_mask)
    # predicted_outputs = tf.reshape(tf.boolean_mask(predicted_outputs, feature_mask),[params.batch_size,params.window_size//18,-1])
    feature_mask = labels > 1e-6
    predicted_outputs = tf.Print(predicted_outputs,[predicted_outputs,tf.math.reduce_max(predicted_outputs),tf.shape(predicted_outputs)],"predicted_outputs-premask",summarize=12,first_n=20)
    labels = tf.Print(labels,[labels,tf.math.reduce_max(labels),tf.shape(labels)],"labels-premask",summarize=12,first_n=20)
    # label_mask = np.full((params.batch_size,params.window_size,params.num_cols), False)
    # label_mask[:,::18,::5]=True
    # label_mask[:,::18,1::5]=True
    # labels = tf.reshape(tf.boolean_mask(labels, label_mask),[params.batch_size,params.window_size//18,-1])
    predicted_outputs = tf.boolean_mask(predicted_outputs, feature_mask)
    labels            = tf.boolean_mask(labels, feature_mask)


    # predicted_outputs = tf.boolean_mask(predicted_outputs, feature_mask)
    # labels = tf.boolean_mask(labels, feature_mask)
    labels = tf.Print(labels,[labels,tf.math.reduce_max(labels),tf.shape(labels)],"labels",summarize=12,first_n=20)
    predicted_outputs = tf.Print(predicted_outputs,[predicted_outputs,tf.math.reduce_max(predicted_outputs),tf.shape(predicted_outputs)],"predicted_outputs",summarize=12,first_n=20)

    losses = tf.square(predicted_outputs-labels)
    # weights = tf.trainable_variables()
    # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in weights])*0.001

    # predicted_outputs = tf.Print(predicted_outputs,[predicted_outputs,tf.shape(predicted_outputs)],"predicted_outputs-for-loss",summarize=10,first_n=10)
    losses = tf.Print(losses,[losses,tf.shape(losses)],"losses",summarize=10,first_n=10)
    # losses = tf.square(predicted_outputs-labels)#tf.losses.mean_squared_error(predicted_outputs,labels)
    # mask = tf.sequence_mask(sentence_lengths)
    # timestep_mask = np.full((32,120,5), True)
    # losses = tf.boolean_mask(losses, timestep_mask)

    loss = tf.reduce_mean(losses)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
    TINY = 1e-6
    mape = tf.reduce_mean(
        tf.clip_by_value(
        tf.abs((labels - predicted_outputs)/ (labels+TINY))
        ,0,1))
    accuracy = 1 - mape

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)#tf.train.AdamOptimizer(params.learning_rate)#RMSPropOptimizer(0.001)#AdamOptimizer(params.learning_rate)#tf.train.GradientDescentOptimizer(0.001)#
        global_step = tf.train.get_or_create_global_step()
        # train_op = optimizer.minimize(loss, global_step=global_step)

        gradients, variables = zip(*optimizer.compute_gradients(loss))
        # for g, v in zip(gradients,variables):
        #   tf.summary.histogram(v.name, v)
        #   tf.summary.histogram(v.name + '_grad', g)
        # def ClipIfNotBad(grad):
            # return tf.where(tf.is_finite(grad), grad, 0.5*tf.ones_like(grad))
            # return tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad)
        #     tf.cond(tf.reduce_sum(tf.cast(tf.math.logical_not(tf.is_finite(grad)),tf.float32))>0,lambda: grad,lambda: 0.5*tf.ones_like(grad))#tf.is_finite(tf.reduce_sum(grad)
        # gradients = [ClipIfNotBad(grad) for grad in gradients]
        # gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients]

        # gradients = tf.where(tf.is_nan(gradients), tf.zeros_like(gradients), gradients)
        # def replace_none_with_zero(l):
        #     return [0 if i==None else i for i in l]

        # gradients = [0.0 if i==None else i for i in gradients]
        # gradients = [tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]

        # gradients = tf.Print(gradients,[gradients,tf.math.reduce_mean(gradients)],"gradients",summarize=10,first_n=10)

        gradients, _ = tf.clip_by_global_norm(gradients, 5.0) #
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        #clip by value
        # grads = optimizer.compute_gradients(loss)
        # clipped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        # train_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    #tf.metrics.accuracy(labels=labels, predictions=predictions),
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.mean(accuracy),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['labels'] = labels
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['predictions'] = predicted_outputs
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()


    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
