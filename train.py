"""Train the model"""
#train.py to change from train to evaluate

import argparse
import logging
import os

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.input_fn import load_dataset_from_csv
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(42)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    params.update(json_path)
    # num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    # overwritting = model_dir_has_best_weights and args.restore_dir is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Get paths for vocabularies and dataset
    # path_words = os.path.join(args.data_dir, 'words.txt')
    # path_tags = os.path.join(args.data_dir, 'tags.txt')
    # path_train_sentences = os.path.join(args.data_dir, 'train/sentences.txt')
    # path_train_labels = os.path.join(args.data_dir, 'train/labels.txt')
    # path_eval_sentences = os.path.join(args.data_dir, 'dev/sentences.txt')
    # path_eval_labels = os.path.join(args.data_dir, 'dev/labels.txt')

    # Load Vocabularies
    # words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=num_oov_buckets)
    # tags = tf.contrib.lookup.index_table_from_file(path_tags)

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    # train_x = load_dataset_from_csv(filenames = ["data/train/SH1N30s2-in-1.csv"])#["data/SH1N30s2c.csv"])
    train_x = load_dataset_from_csv(filenames = ["data/train/payne/payne_train_in_a_1.4_pcr_90.0_vf_110.0.csv"])
    train_y = load_dataset_from_csv(filenames = ["data/train/payne/payne_train_out_a_1.4_pcr_90.0_vf_110.0.csv"])#"data/train/SH1N30s2-out-1.csv"])#["data/SH1N30s2c.csv"])#["data/labels/SH1N30s2c.csv"])
    # train_labels = load_dataset_from_csv()#load_dataset_from_csv(path_train_labels)
    eval_x = load_dataset_from_csv(filenames = ["data/test/payne/payne_test_in_a_1.4_pcr_90.0_vf_110.0.csv"])#"data/test/SH1N30s2-in-1.csv"])#["data/eval/SH1N30s2c-in.csv"])#load_dataset_from_csv(path_eval_sentences)
    eval_y = load_dataset_from_csv(filenames = ["data/test/payne/payne_test_out_a_1.4_pcr_90.0_vf_110.0.csv"])#"data/test/SH1N30s2-out-1.csv"])#["data/eval/SH1N30s2c-out.csv"])
    # eval_labels = load_dataset_from_csv()

    # Specify other parameters for the dataset and the model
    params.eval_size = params.dev_size

#     p_cr[[198.91478][198.91568][198.917023][198.919113][198.921753][198.925049][198.928757][198.933105][198.937988][198.943604]...][198.881866]
# stat_speed[[10.0828867 10.0831852 10.0842876 10.0847178 10.0842752 10.0829487 10.0807686 10.0778112 10.0743017 10.0707045...]...][87.5445099]
# v_f[[9.49845886][9.15258789][8.90024][8.71338177][8.59364][8.51922226][8.49839497][8.51756382][8.58668518][8.70912075]...][43.092617]
# a[[2.99534559][2.99534583][2.99535036][2.99535751][2.995368][2.9953804][2.9953959][2.99541402][2.99543619][2.99546194]...][2.99563766]
# boundry[[[10000 0.0372472554][3934.96216 42.9395828]][[10000 0.0383180901][3937.93042 43.0135765]][[10000 0.0392575823]]...][3435.42896]
# p_cr[[198.938522][198.940201][198.942551][198.945389][198.948792][198.952591][198.956909][198.961731][198.967316][198.973633]...][198.912537]
# stat_speed[[9.48225784 9.48253632 9.48380089 9.48442936 9.48426628 9.48327541 9.48146439 9.47888374 9.4757061 9.47231483...]...][88.6305923]
# boundry[[[10000 0.0381684676][3958.17871 43.0536118]][[10000 0.0391164869][3959.71094 43.1307869]][[10000 0.0399727598]]...][3430.82104]
# v_f[[9.03243542][8.77550888][8.58525][8.46234226][8.38595486][8.36284637][8.38093185][8.44868374][8.56938744][8.74094]...][43.0496292]
# a[[2.99536753][2.99537325][2.99538159][2.9953928][2.99540567][2.99542141][2.99543977][2.99546146][2.99548745][2.99551725]...][2.99567032]
# p_cr[[198.955154][198.957687][198.960632][198.964066][198.967896][198.972214][198.976974][198.982483][198.988739][198.995743]...][198.93692]
# stat_speed[[9.01641083 9.01664448 9.01807404 9.01889896 9.0189743 9.01827335 9.01678658 9.01454067 9.01166 9.00846195...]...][89.541748]
# boundry[[[10000 0.039043095][3985.56616 43.1513176]][[10000 0.0399001949][3986.0481 43.2289734]][[10000 0.0406362899]]...][3425.51367]
# boundry[[[10000 0.0398645699][4014.59961 43.236042]][[10000 0.0405981913][4013.93335 43.313179]][[10000 0.0412503853]]...][3421.69482]
# boundry[[[10000 0.0405803509][4043.34814 43.3117676]][[10000 0.0412281][4041.9126 43.3867]][[10000 0.0417642891]]...][3418.10522]

    params.buffer_size = 6000#params.train_size # buffer size for shuffling
    params.restore_dir= "experiments/best_weights"#"experiments/last_weights"#None#args.restore_dir#"experiments/best_weights"#None#
    # params.id_pad_word = words.lookup(tf.constant(params.pad_word))
    # params.id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_x,train_y, params)

    eval_inputs = input_fn('eval', eval_x,eval_y, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    # train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_dir)
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, params.restore_dir)
