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
        # Get word embeddings for each token in the sentence
        # embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
        #         shape=[params.vocab_size, params.embedding_size])
        # sentence = tf.nn.embedding_lookup(embeddings, sentence)

        # Apply LSTM over the embeddings
        # lstm_cell = tf.nn.rnn_cell.LSTMCell(params.lstm_num_units,use_peepholes=True,cell_clip=3.0)
        # lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(params.lstm_num_units)
        lstm_cell = ntfCell(params.lstm_num_units)#,use_peepholes=True,cell_clip=3.0)
        # lstm_cell = LSTMCell2(params.lstm_num_units)#,use_peepholes=True,cell_clip=3.0)


        rnn_outputs, rnn_states  = tf.nn.dynamic_rnn(lstm_cell, input_batch, dtype=tf.float32)

        # # Compute logits from the output of the LSTM
        # logits = tf.layers.dense(output, params.number_of_tags)
        # project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
        # an extra layer here.
        # final_projection = lambda x: tf.layers.dense(x,1)# params.rnn_output_size)
        # apply projection to every timestep.
        # predicted_outputs = tf.map_fn(final_projection, rnn_outputs)
        predicted_outputs =  tf.layers.dense(rnn_outputs, 256,activation='relu')
        predicted_outputs =  tf.layers.dense(predicted_outputs, params.rnn_output_size)

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
    # feature_mask = np.full((32,120,90), False)
    # feature_mask[:,:,:]=True#20*2:25*2:2] = True
    # losses = tf.boolean_mask(losses, mask)
    # predicted_outputs = tf.reshape(tf.boolean_mask(predicted_outputs, feature_mask),[32,120,5])
    # labels = tf.reshape(tf.boolean_mask(labels, feature_mask),[32,120,5])


    # predicted_outputs = tf.boolean_mask(predicted_outputs, feature_mask)
    # labels = tf.boolean_mask(labels, feature_mask)


    losses = tf.square(predicted_outputs-labels)

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
        optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)#RMSPropOptimizer(0.001)#AdamOptimizer(params.learning_rate)#tf.train.GradientDescentOptimizer(0.001)#
        global_step = tf.train.get_or_create_global_step()
        # train_op = optimizer.minimize(loss, global_step=global_step)

        gradients, variables = zip(*optimizer.compute_gradients(loss))
        def ClipIfNotBad(grad):
            return tf.where(tf.is_finite(grad), grad, 0.5*tf.ones_like(grad))
            return tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad)
            tf.cond(tf.reduce_sum(tf.cast(tf.math.logical_not(tf.is_finite(grad)),tf.float32))>0,lambda: grad,lambda: 0.5*tf.ones_like(grad))#tf.is_finite(tf.reduce_sum(grad)
        gradients = [tf.clip_by_value(ClipIfNotBad(grad), -1., 1.) for grad in gradients]
        # gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients]

        # gradients = tf.where(tf.is_nan(gradients), tf.zeros_like(gradients), gradients)

        gradients, _ = tf.clip_by_global_norm(gradients, 3.0) #
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
