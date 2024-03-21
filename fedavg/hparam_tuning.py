import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import os
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
    nargs='+',
    required=True,
    help='Learning rate for training.')

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
    '-sd',
    '--seed',
    action='store',
    type=int,
    required=True,
    help='Seed for sampling clients and their budgets.'
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
    '-sm',
    '--save_model',
    action='store',
    type=bool,
    default=False,
    help='Set to True to save final global model to log directory. Default is False.'
)

args = my_parser.parse_args()
for k,v in vars(args).items():
    print(k, ":", v)

# Set args from arg parser ---------------
B = args.batch_size
log_dir = args.logdir
dset = args.dataset
run_fl_hparams = {'batch_size':B}
train_dir = args.training_dir
test_dir = args.testing_dir
lower_bound = args.lower_bound
upper_bound = args.upper_bound
    
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

# Hyperparameter tuning code
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete(args.learning_rate))
METRIC_ACCURACY = 'accuracy'
METRIC_BA_ROUND_NUM = 'best_accuracy_round_num'
METRIC_TEST_LOSS = 'test_loss'
METRIC_TEST_ACCURACY = 'test_accuracy'
METRIC_TRAIN_LOSS = 'train_loss'
METRIC_TRAIN_ACCURACY = 'train_sparse_categorical_accuracy'

# Logging structure
# log_dir
#   + hparams
#   + session_1
#       /tb
#       /params.txt
#   + session_2
#       /tb
#       /params.txt

with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=[HP_LEARNING_RATE],
        metrics=[
            hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
            hp.Metric(METRIC_BA_ROUND_NUM, display_name='Best accuracy #R.N'),
            hp.Metric(METRIC_TEST_LOSS, display_name='Test Loss'),
            hp.Metric(METRIC_TEST_ACCURACY, display_name='Test Accuracy'),
            hp.Metric(METRIC_TRAIN_LOSS, display_name='Train Loss'),
            hp.Metric(METRIC_TRAIN_ACCURACY, display_name='Train Accuracy')
        ]
    )

def run(run_dir, hparams):
    global args, dataset, preprocess, central_test_dataset, evaluate, check_stopping_criteria, TC, run_fl_hparams
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        tensorboard_logdir = run_dir

        with open(os.path.join(run_dir, 'params.txt'), 'w') as f:
            for k,v in vars(args).items():
                f.write(str(k) + " : " + str(v) + "\n")

        # For reproducible client selection and budget sampling
        rng = np.random.default_rng(args.seed)

        # call run_fl
        best_accuracy, best_accuracy_round_num, _, _, _, _ = run_fl(
            rng,
            dataset,
            preprocess,
            central_test_dataset,
            evaluate,
            check_stopping_criteria,
            run_fl_hparams, 
            tensorboard_logdir,
            lower_bound,  # lower bound for client budgets
            upper_bound,  # upper bound for client budgets
            TC,
            num_clients=args.num_clients,
            fixed_rounds=args.fixed_rounds,
            evaluate_every=args.evaluate_every,
            lr_schedule=lambda n: hparams[HP_LEARNING_RATE],
            n_guesses=args.num_guesses,
            model_weights_file=args.model_weights_file
        )
        # write logs
        tf.summary.scalar(METRIC_ACCURACY, best_accuracy, step=1)
        tf.summary.scalar(METRIC_BA_ROUND_NUM, best_accuracy_round_num, step=1)

session_num = 0

for lr in HP_LEARNING_RATE.domain.values:
    hparams = {
        HP_LEARNING_RATE: lr
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run(log_dir + '/' + run_name, hparams)
    session_num += 1