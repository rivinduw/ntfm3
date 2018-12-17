"""Create the input data pipeline using `tf.data`"""
#input_fn stores the input data pipeline

import tensorflow as tf

numCols = 5*2
#max_vals = [46,24,136,23.8333,136,23.8333,175,19.2778,97,27.5,63,13.6667,454,24.5,899,37.25,850,67,1262,57.8571,2064,56.9231,87,55.6667,804,66,1362,67.3333,472,83.6667,560,68.3333,563,74.1667,563,74.1667,952,70.8333,437,60.9583,1275,61.2222,884,78.3333,394,80.5,394,80.5,189,84,325,49.4167,616,44.5,651,60,718,59,1546,54.1667,1546,60.1667,796,77.25,1356,67.3333,772,69.1667,162,80.6667,162,80.6667,485,55,307,56.525,293,51.75,271,56.65,308,56.65,299,59.2857,250,64.8,126,94,126,94]

# max_vals = [ 4462.11559904,    53.7349221 ,  4423.03294541,    73.53982445,\
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
max_vals = [22520.0,188.0,38080.0,317.5,17480.0,145.5,51000.0,425.0,35360.0,295.0]

max_values=tf.convert_to_tensor(max_vals)#[1000.0,50.0]*45)
# future_step = 1

def load_dataset_from_csv(filenames = ["data/SH1N30s2.csv"]):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line

    Returns:
        dataset: (tf.Dataset) yielding list of values at eachtimestep
    """

    # Creates a dataset that reads all of the records from CSV files, with headers,
    #  extracting float data from 90 float columns ater the first datetime column
    record_defaults = [[0.0]] * numCols  # Only provide defaults for the selected columns
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=list(range(1, numCols+1)))


    def parser(*x):
        """The output from the CsvDataset is wierd (has a separate tensor for each feature?).
            Converting it to a tensor makes it of size numFeatures
            Normalizing the features is also done here.
        """
        x = tf.convert_to_tensor(x)
        # 90 columns, the even columns are volumes and odd are occupancies
        # max value for volume was around 2000veh-ish and 100% for occupancies
        #TODO: remove hardcoded values. could pass list of max values for each feature
        max_values=tf.convert_to_tensor(max_vals)#[1000.0,50.0]*45)
        # x = tf.div(x,max_values) #divide by max makes x [0,1] (usually no negatives)
        return x
    dataset = dataset.map(parser)


    # def label_parser(*y):
    #     """The output from the CsvDataset is wierd (has a separate tensor for each feature?).
    #         Converting it to a tensor makes it of size numFeatures
    #         Normalizing the features is also done here.
    #     """
    #     y = tf.convert_to_tensor(y)
    #     # 90 columns, the even columns are volumes and odd are occupancies
    #     # max value for volume was around 2000veh-ish and 100% for occupancies
    #     #TODO: remove hardcoded values. could pass list of max values for each feature
    #     # max_values=tf.convert_to_tensor(max_vals)#[1000.0,50.0]*45)
    #     # x = tf.div(x,max_values) #divide by max makes x [0,1] (usually no negatives)
    #     return y[future_step:]
    # labels = dataset.map(label_parser)
    # dataset = dataset.map(lambda *x: tf.div(tf.convert_to_tensor(x),100))
    # dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size=120, window_shift=10))

    # # Load txt file, one example per line
    # dataset = tf.data.TextLineDataset(path_txt)
    # # Convert line into list of tokens, splitting by white space
    # dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # # Lookup tokens to return their ids
    # dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))

    return dataset#inputs,labels#


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
    # labels = inputs.map(tf.manip.roll(inputs,shift=1, axis=0))
    dataset = tf.data.Dataset.zip((inputs, labels))

    # Create batches and pad the sentences of different length


    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size=120, window_shift=1))
    #tf.data.Dataset.window(size=window_size, shift=window_shift, stride=window_stride).flat_map(lambda x: x.batch(window.size))

    dataset = (dataset
        # .apply(tf.contrib.data.sliding_window_batch(window_size=120, window_shift=1))
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
        'label_batch': label_batch,#max_values*label_batch,#max_values*input_batch,
        'iterator_init_op': init_op
    }

    return inputs
