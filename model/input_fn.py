"""Create the input data pipeline using `tf.data`"""
#input_fn stores the input data pipeline

import tensorflow as tf

filenames = ["data/SH1N30s2.csv"]
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


    def parser(*x):
        """The output from the CsvDataset is wierd (has a separate tensor for each feature?).
            Converting it to a tensor makes it of size numFeatures
            Normalizing the features is also done here.
        """
        x = tf.convert_to_tensor(x)
        # 90 columns, the even columns are volumes and odd are occupancies
        # max value for volume was around 2000veh-ish and 100% for occupancies
        #TODO: remove hardcoded values. could pass list of max values for each feature
        max_values=tf.convert_to_tensor([2000.0,100.0]*45)
        x = tf.div(x,max_values) #divide by max makes x [0,1] (usually no negatives)
        return x
    dataset = dataset.map(parser)
    # dataset = dataset.map(lambda *x: tf.div(tf.convert_to_tensor(x),100))
    # dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size=120, window_shift=10))

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
    dataset = inputs #tf.data.Dataset.zip((inputs, labels))

    # Create batches and pad the sentences of different length
    dataset = (dataset
        # .apply(tf.contrib.data.sliding_window_batch(window_size=120, window_shift=10))
        .shuffle(buffer_size=buffer_size)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    dataset = dataset.batch(params.batch_size)

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    input_batch = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'input_batch': input_batch,
        'label_batch': input_batch,
        'iterator_init_op': init_op
    }

    return inputs
