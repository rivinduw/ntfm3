"""Create the input data pipeline using `tf.data`"""
#input_fn stores the input data pipeline

import tensorflow as tf

numCols = 90
#max_vals = [46,24,136,23.8333,136,23.8333,175,19.2778,97,27.5,63,13.6667,454,24.5,899,37.25,850,67,1262,57.8571,2064,56.9231,87,55.6667,804,66,1362,67.3333,472,83.6667,560,68.3333,563,74.1667,563,74.1667,952,70.8333,437,60.9583,1275,61.2222,884,78.3333,394,80.5,394,80.5,189,84,325,49.4167,616,44.5,651,60,718,59,1546,54.1667,1546,60.1667,796,77.25,1356,67.3333,772,69.1667,162,80.6667,162,80.6667,485,55,307,56.525,293,51.75,271,56.65,308,56.65,299,59.2857,250,64.8,126,94,126,94]

max_vals = [ 1434.74750643,    18.30139494,  1429.74008344,    17.48155306,\
        1426.702155  ,    17.99523301,  1424.37520692,    18.08119266,\
        1422.62415457,    18.07461953,  1421.35611114,    18.05625932,\
        1420.45478819,    18.04031743,  1419.90337358,    18.03026502,\
        1419.52384034,    18.02317216,  1419.20911214,    18.01715104,\
        1418.95221455,    18.01225176,  1418.70013052,    18.00769527,\
        1418.43693812,    18.00269478,  1418.14779313,    17.99711155,\
        1417.81579209,    17.99103296,  1417.44840569,    17.98407964,\
        1417.06875011,    17.97687155,  1416.64375747,    17.96918448,\
        1416.22762397,    17.96095446,  1415.7796153 ,    17.95286074,\
        1415.32413225,    17.9440461 ,  1414.86735724,    17.93563274,\
        1414.38253656,    17.92676427,  1413.92731718,    17.91788703,\
        1413.4421255 ,    17.90913208,  1412.97392148,    17.89990094,\
        1412.50099183,    17.89137473,  1412.01763006,    17.88237556,\
        1411.56109772,    17.8736535 ,  1411.08129417,    17.86500379,\
        1410.62835788,    17.85608305,  1410.16940487,    17.84781971,\
        1409.70701819,    17.8391717 ,  1409.26995687,    17.83088746,\
        1408.81330982,    17.82266766,  1408.3853254 ,    17.81425268,\
        1407.95186431,    17.8064622 ,  1407.51721571,    17.79833338,\
        1407.10663076,    17.79058055,  1406.67835256,    17.78288625,\
        1406.27857516,    17.77504053,  1405.87307208,    17.76776747,\
        1405.46838831,    17.76017997,  1405.08484042,    17.7529909 ,\
        1404.7018785 ,    17.75292314]
        
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
        max_values=tf.convert_to_tensor(max_vals)#[1000.0,50.0]*45)
        x = tf.div(x,max_values) #divide by max makes x [0,1] (usually no negatives)
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
        'label_batch': max_values*label_batch,#max_values*input_batch,
        'iterator_init_op': init_op
    }

    return inputs
