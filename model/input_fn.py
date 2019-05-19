"""Create the input data pipeline using `tf.data`"""
#input_fn stores the input data pipeline

import tensorflow as tf
# max_vals = [22520.0,188.0,38080.0,317.5,17480.0,145.5,51000.0,425.0,35360.0,295.0]

def load_dataset_from_csv(filenames = ["data/SH1N30s2.csv"],params=None):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line

    Returns:
        dataset: (tf.Dataset) yielding list of values at eachtimestep
    """


    num_cols = int(params.num_cols)
    max_values=tf.convert_to_tensor(params.max_vals)
    # Creates a dataset that reads all of the records from CSV files, with headers,
    #  extracting float data from 90 float columns ater the first datetime column
    record_defaults = [[0.0]] * num_cols  # Only provide defaults for the selected columns
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=list(range(1, num_cols+1)))

    def parser(*x):
        """The output from the CsvDataset is wierd (has a separate tensor for each feature?).
            Converting it to a tensor makes it of size numFeatures
            Normalizing the features is also done here.
        """
        x = tf.convert_to_tensor(x)
        #BUG: divide by zero
        # x = tf.div(x+1e-12,max_values+1e-6) #divide by max makes x [0,1] (usually no negatives)
        return x
    dataset = dataset.map(parser)

    return dataset


def input_fn(mode, inputs, labels, params):
    """Input function for NER

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        inputs: (tf.Dataset) yielding list of ids of words
        datasets: (tf.Dataset) yielding list of ids of tags
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1

    # Zip the inputs and labels together
    dataset = tf.data.Dataset.zip((inputs, labels))

    # Create batches and pad the sentences of different length
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size=params.window_size, window_shift=params.window_shift)) #360*10s = 1hour
    #tf.data.Dataset.window(size=window_size, shift=window_shift, stride=window_stride).flat_map(lambda x: x.batch(window.size))

    if mode=='eval':
        dataset = (dataset
            .repeat()
            .batch(params.batch_size)
            .prefetch(1)
        )
    else:
        dataset = (dataset
            .shuffle(buffer_size=buffer_size)
            .batch(params.batch_size)
            .repeat()
            .prefetch(1)  # make sure you always have one batch ready to serve
        )


    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    label_window = params.label_window
    # Query the output of the iterator for input to the model
    input_batch, label_batch = iterator.get_next()
    new_input_batch = tf.zeros_like(input_batch[:,label_window:,:])
    # new_label_batch = tf.zeros_like(label_batch[:,3600:,:])

    new2_input_batch = tf.concat([input_batch[:,:label_window,:],new_input_batch], axis=1)
    # new2_label_batch = tf.concat([label_batch[:,:3600,:],new_label_batch], axis=1)

    new2_input_batch.set_shape([input_batch.get_shape()[0], input_batch.get_shape()[1],input_batch.get_shape()[2]])
    # new2_label_batch.set_shape([label_batch.get_shape()[0], label_batch.get_shape()[1],label_batch.get_shape()[2]])

    input_batch = new2_input_batch
    # label_batch = new2_label_batch


    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'input_batch': input_batch,
        'label_batch': label_batch,#max_values*label_batch i fyuo want unscaled
        'iterator_init_op': init_op
    }

    return inputs
