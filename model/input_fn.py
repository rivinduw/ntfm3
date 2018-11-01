"""Create the input data pipeline using `tf.data`"""
#input_fn stores the input data pipeline

import tensorflow as tf

filenames = ["../data/SH1N30s2.csv"]
numCols = 90

def load_dataset_from_csv(path_txt="../data/"):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line

    Returns:
        dataset: (tf.Dataset) yielding list of values at eachtimestep
    """

    # Creates a dataset that reads all of the records from CSV files, with headers,
    #  extracting float data from 90 float columns ater the first datetime column
    record_defaults = [[0.0]] * numCols  # Only provide defaults for the selected columns
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=list(range(1, 91)))

    dataset.apply(tf.contrib.data.sliding_window_batch(window_size=120, window_shift=10))

    # # Load txt file, one example per line
    # dataset = tf.data.TextLineDataset(path_txt)
    # # Convert line into list of tokens, splitting by white space
    # dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # # Lookup tokens to return their ids
    # dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))

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

    # Zip the sentence and the labels together
    dataset = tf.data.Dataset.zip((inputs, labels))

    # Create batches and pad the sentences of different length
    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    input_batch = iterator.get_next()
    init_op = iterator.initializer

    print(input_batch)

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'input_batch': input_batch[:-1],
        'label_batch': input_batch[1:],
        'iterator_init_op': init_op
    }

    return inputs
