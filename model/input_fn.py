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


    num_cols = params.num_cols
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
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size=params.window_size, window_shift=1)) #360*10s = 1hour
    #tf.data.Dataset.window(size=window_size, shift=window_shift, stride=window_stride).flat_map(lambda x: x.batch(window.size))

    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .batch(params.batch_size)
        .prefetch(2)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    input_batch, label_batch = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'input_batch': input_batch,
        'label_batch': label_batch,#max_values*label_batch i fyuo want unscaled
        'iterator_init_op': init_op
    }

    return inputs
