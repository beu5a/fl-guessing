# Environment for running simulations
```
# Inside efficient-federated-learning folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
**Note:** There has been an update to `requirements.txt`. Please delete `venv` and redo the above step.

# Getting the data from LEAF
```
git clone https://github.com/TalwalkarLab/leaf.git
sudo apt-get install unzip
# cd into leaf/data/shakespeare
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 --smplseed 10 --spltseed 10

# cd into leaf/data/femnist (takes ~2 hrs)
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample --ssmplseed 10 --spltseed 10

# cd into leaf/data/celeba
./preprocess.sh -s niid --sf 1.0 -t sample --smplseed 10 --spltseed 10 -k 5
```

# Running gradient guessing
```
# cd into grad_guessing/
python main.py <specify params>

  -d {femnist,shakespeare,celeba,synthetic,sent140,reddit}, --dataset {femnist,shakespeare,celeba,synthetic,sent140,reddit}
                        Dataset on which to run experiment.
  -traindir TRAINING_DIR, --training_dir TRAINING_DIR
                        Absolute path to the directory containing training
                        data.
  -testdir TESTING_DIR, --testing_dir TESTING_DIR
                        Absolute path to the directory containing testing
                        data.
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for training.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training.
  -lb LOWER_BOUND, --lower_bound LOWER_BOUND
                        Lower bound for budget for U(lb, up)
  -up UPPER_BOUND, --upper_bound UPPER_BOUND
                        Uppper bound for budget for U(lb, up)
  -l LOGDIR, --logdir LOGDIR
                        Path to directory for logging. Creates one if not
                        exists.
  -c {round_robin,fastest}, --selection_scheme {round_robin,fastest}
                        Selection for client selection based on computational
                        budget.
  -n NUM_CLIENTS, --num_clients NUM_CLIENTS
                        Number of clients to be selected in every round.
  -f FIXED_ROUNDS, --fixed_rounds FIXED_ROUNDS
                        Number of rounds to run if running for fixed rounds.
  -ee EVALUATE_EVERY, --evaluate_every EVALUATE_EVERY
                        Frequency of evaluation on test set.
  -g NUM_GUESSES, --num_guesses NUM_GUESSES
                        Total number of guesses to be made. Values: max,
                        b<int> or <int>.
  -sm SAVE_MODEL, --save_model SAVE_MODEL
                        Set to True to save final global model to log
                        directory. Default is False.
  -scf SAMPLED_CLIENTS_FILE, --sampled_clients_file SAMPLED_CLIENTS_FILE
                        Points to the file containing sampled clients.
  -sbf SAMPLED_BUDGETS_FILE, --sampled_budgets_file SAMPLED_BUDGETS_FILE
                        Points to the file containing sampled budgets.
  -mwf MODEL_WEIGHTS_FILE, --model_weights_file MODEL_WEIGHTS_FILE
                        Points to the file containing model weights for same
                        initialisation.
```

# Running gradient measurements
```
# Stay in root directory i.e efficient-federated-learning
# note - no '.py' at end of 'main'
python -m gradient_measurements.main <params> 

-d {femnist,shakespeare}, --dataset {femnist,shakespeare}
                    Dataset on which to run experiment.
-b {10,20,32,64}, --batchsize {10,20,32,64}
                    Size of the batch for dataset.
-e EPOCHS, --epochs EPOCHS
                    Size of the batch for dataset.
-o {rmsprop,sgd}, --optimiser {rmsprop,sgd}
                    Optimisation algorithm for the learning task.
-l LOGDIR, --logdir LOGDIR
                    Path to directory for logging. Creates one if not exists.
-traindir TRAINING_DIR, --training-dir TRAINING_DIR
                    Absolute path to the directory containing training data.
-testdir TESTING_DIR, --testing-dir TESTING_DIR
                    Absolute path to the directory containing testing data.
```

# Running tensorboard
```
tensorboard --logdir <log_folder/tb>
```
