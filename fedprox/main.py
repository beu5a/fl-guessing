import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import builtins
from models import *
from data_utils import *
from stopping import *
from evaluate import *

my_parser = argparse.ArgumentParser()

my_parser.add_argument(
    '-d',
    '--dataset',
    action='store',
    type=str,
    choices=['femnist', 'shakespeare', 'celeba', 'synthetic', 'sent140', 'reddit'],
    default='shakespeare',
    help='Dataset on which to run experiment.')

my_parser.add_argument(
    '-traindir',
    '--training_dir',
    action='store',
    type=str,
    required=True,
    help='Absolute path to the directory containing training data.')

my_parser.add_argument(
    '-testdir',
    '--testing_dir',
    action='store',
    type=str,
    required=True,
    help='Absolute path to the directory containing testing data.')

my_parser.add_argument(
    '-r',
    '--learning_rate',
    action='store',
    type=float,
    required=True,
    help='Learning rate for training.')

my_parser.add_argument(
    '-mu',
    '--mu',
    action='store',
    type=float,
    default=0.01,
    help='FedProx regularizer.')

my_parser.add_argument(
    '-b',
    '--batch_size',
    action='store',
    type=int,
    required=True,
    help='Batch size for training.')

my_parser.add_argument(
    '-lb',
    '--lower_bound',
    action='store',
    type=int,
    required=True,
    help='Lower bound for budget for  U(lb, up)')

my_parser.add_argument(
    '-up',
    '--upper_bound',
    action='store',
    type=int,
    required=True,
    help='Uppper bound for budget for  U(lb, up)')

my_parser.add_argument(
    '-l',
    '--logdir',
    action='store',
    type=str,
    default='./logs',
    required=True,
    help='Path to directory for logging. Creates one if not exists.')

my_parser.add_argument(
    '-n',
    '--num_clients',
    action='store',
    type=int,
    default=20,
    help='Number of clients to be selected in every round.')

my_parser.add_argument(
    '-f',
    '--fixed_rounds',
    action='store',
    type=int,
    help='Number of rounds to run if running for fixed rounds.')

my_parser.add_argument(
    '-ee',
    '--evaluate_every',
    action='store',
    type=int,
    default=3,
    help='Frequency of evaluation on test set.')

my_parser.add_argument(
    '-g',
    '--num_guesses',
    action='store',
    type=str,
    required=True,
    help='Total number of guesses to be made. Values: max, b<int> or <int>.')

my_parser.add_argument(
    '-sm',
    '--save_model',
    action='store',
    type=bool,
    default=False,
    help='Set to True to save final global model to log directory. Default is False.'
)

my_parser.add_argument(
    '-mwf',
    '--model_weights_file',
    action='store',
    type=str,
    default=None,
    help='Points to the file containing model weights for same initialisation.'
)

my_parser.add_argument(
    '-sd',
    '--seed',
    action='store',
    type=int,
    required=True,
    help='Seed for sampling clients and their budgets.'
)

args = my_parser.parse_args()
for k,v in vars(args).items():
    print(k, ":", v)

# Set args from arg parser ---------------
eta = args.learning_rate
mu = args.mu
B = args.batch_size
lower_bound = args.lower_bound
upper_bound = args.upper_bound
log_dir = args.logdir
dset = args.dataset
hparams = {'batch_size':B}
train_dir = args.training_dir
test_dir = args.testing_dir
    
# Set functions/fields based on the dataset ---------------
check_stopping_criteria = None
dataset = None
preprocess = None
central_test_dataset = None
evaluate = None
builtins.model_fn = None
builtins.keras_model_fn = None
TC = None

#TC = 660 (Total clients for non-iid)
if(dset == 'shakespeare'):
    # Make this global across all modules
    builtins.model_fn = tff_shakespeare_model_fn
    builtins.keras_model_fn = get_stacked_lstm

    TEST_BATCH_SIZE = 4096
    check_stopping_criteria = check_stopping_criteria_shakespeare
    dataset = ShakespeareData(train_dir, test_dir)
    preprocess = preprocess_shakespeare
    central_test_dataset = dataset.create_test_dataset_for_all_clients().map(to_ids).batch(TEST_BATCH_SIZE)    
    evaluate = evaluate_shakespeare

#TC = 3597 (Total clients for non-iid)
elif(dset == 'femnist'):
    # Make this global across all modules
    builtins.model_fn = tff_femnist_model_fn
    builtins.keras_model_fn = get_femnist_cnn

    TEST_BATCH_SIZE = 2048
    check_stopping_criteria = check_stopping_criteria_femnist
    dataset = FemnistData(train_dir, test_dir)
    preprocess = preprocess_femnist
    central_test_dataset = dataset.create_test_dataset_for_all_clients().batch(TEST_BATCH_SIZE).map(femnist_batch_format_fn)
    evaluate = evaluate_femnist

# TC = 9343 (Total clients for non-iid)
elif(dset == 'celeba'):
    # Make this global across all modules
    builtins.model_fn = tff_celeba_model_fn
    builtins.keras_model_fn = get_celeba_cnn

    TEST_BATCH_SIZE = 1024
    check_stopping_criteria = check_stopping_criteria_celeba
    images_dir = os.path.join(train_dir, '../raw/img_align_celeba')
    dataset = CelebaData(train_dir, test_dir, images_dir)
    preprocess = preprocess_celeba
    central_test_dataset = dataset.create_test_dataset_for_all_clients().batch(TEST_BATCH_SIZE).map(celeba_batch_format_fn)
    evaluate = evaluate_celeba

# TC = 1000 (Total clients for non-iid)
elif(dset == 'synthetic'):
    # Make this global across all modules
    builtins.model_fn = tff_synthetic_model_fn
    builtins.keras_model_fn = get_synthetic_perceptron

    TEST_BATCH_SIZE = 2048
    check_stopping_criteria = check_stopping_criteria_synthetic
    dataset = SyntheticData(train_dir, test_dir)
    preprocess = preprocess_synthetic
    central_test_dataset = dataset.create_test_dataset_for_all_clients().batch(TEST_BATCH_SIZE).map(synthetic_batch_format_fn)
    evaluate = evaluate_synthetic

#TC = 67107 (Total clients for non-iid)
elif(dset == 'sent140'):
    embedding_dir = os.path.join(train_dir, '../../../../models/sent140/embs.json')
    with open(embedding_dir, 'r') as f:
        embs = json.load(f)

    # Make this global across all modules
    builtins.id2word = embs['vocab']
    builtins.word2id = {v: k for k,v in enumerate(id2word)}
    builtins.word_emb = np.array(embs['emba'])
    builtins.model_fn = tff_sent140_model_fn
    builtins.keras_model_fn = get_stacked_rnn

    TEST_BATCH_SIZE = 2048
    check_stopping_criteria = check_stopping_criteria_sent140
    dataset = Sent140Data(train_dir, test_dir)
    preprocess = preprocess_sent140
    central_test_dataset = dataset.create_test_dataset_for_all_clients().batch(TEST_BATCH_SIZE).map(sent140_batch_format_fn)    
    evaluate = evaluate_sent140

# TC = -- (Total clients for non-iid)
elif(dset == 'reddit'):
    # Make this global across all modules
    builtins.model_fn = tff_reddit_model_fn
    TEST_BATCH_SIZE = 2048
    check_stopping_criteria = check_stopping_criteria_reddit
    dataset = RedditData(train_dir, test_dir)
    preprocess = preprocess_reddit
    central_test_dataset = dataset.create_test_dataset_for_all_clients().batch(TEST_BATCH_SIZE)
    evaluate = evaluate_reddit

TC = len(dataset.num_samples)

# import fl_setup with globals set ---------------
# runs dynamic tracing with type-checking
from run import run_fl

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_logdir = os.path.join(log_dir, 'tb') 

with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
    for k,v in vars(args).items():
        f.write(str(k) + " : " + str(v) + "\n")

rng = np.random.default_rng(args.seed)

_, _, _, state, all_budgets, all_guesses, train_metrics, test_metrics = run_fl(
    rng,
    dataset,
    preprocess,
    central_test_dataset,
    evaluate,
    check_stopping_criteria,
    hparams, 
    tensorboard_logdir,
    lower_bound,
    upper_bound,
    TC,
    num_clients=args.num_clients,
    fixed_rounds=args.fixed_rounds,
    evaluate_every=args.evaluate_every,
    lr_schedule=lambda _: eta,
    mu_schedule=lambda _: mu,
    n_guesses=args.num_guesses,
    model_weights_file=args.model_weights_file)

# Write train, test csv ---------------
train_accuracies, train_losses, train_index = train_metrics
test_accuracies, test_losses, test_index = test_metrics
train_d = {'train_accuracies':train_accuracies, 'train_losses':train_losses, 'train_index':train_index}
test_d = {'test_accuracies':test_accuracies, 'test_losses':test_losses, 'test_index':test_index}
df_train = pd.DataFrame(data=train_d)
df_test = pd.DataFrame(data=test_d)
df_budgets = pd.DataFrame(all_budgets)
df_guesses = pd.DataFrame(all_guesses)
df_train.to_csv(os.path.join(log_dir, 'train.csv'), index=False)
df_test.to_csv(os.path.join(log_dir, 'test.csv'), index=False)
df_budgets.to_csv(os.path.join(log_dir, 'budgets.csv'), index=False, header=False)
df_guesses.to_csv(os.path.join(log_dir, 'guesses.csv'), index=False, header=False)

# Save model ---------------
if(args.save_model):
    if(dset == 'femnist'):
        keras_model = get_femnist_cnn()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
        )
        keras_model.set_weights(state)
    elif(dset == 'shakespeare'):
        keras_model = get_stacked_lstm()
        keras_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]  
        )
        keras_model.set_weights(state)
    elif(dset == 'celeba'):
        keras_model = get_celeba_cnn()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
        )
        keras_model.set_weights(state)
    elif(dset == 'synthetic'):
        keras_model = get_synthetic_perceptron()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
        )
        keras_model.set_weights(state)
    elif(dset == 'sent140'):
        keras_model = get_stacked_rnn()
        keras_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]  
        )
        keras_model.set_weights(state)
    elif(dset == 'reddit'):
        keras_model = get_reddit_rnn()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
        )
        keras_model.set_weights(state)

    trained_model_path = os.path.join(log_dir, 'trained_model')
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)
    keras_model.save(trained_model_path)