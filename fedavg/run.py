import tensorflow_federated as tff
from fl_setup import *
from data_utils import make_federated_data
from math import floor

# Aggregation weights depend not only on number of samples
# but also on budgets since budgets define how many samples
# get seen.


def get_client_agg_weights(budgets, client_num_samples, batch_size):
    seen_training_samples = []
    tp_i = []
    for i in range(len(budgets)):
        tau_i = budgets[i]
        n_i = client_num_samples[i]
        tau_possible_i = floor(n_i / batch_size)
        tp_i.append(tau_possible_i)
        if(tau_i <= tau_possible_i):
            n_final_i = tau_i * batch_size
        else:
            n_final_i = n_i
        seen_training_samples.append(n_final_i)

    total_seen_training_samples = sum(seen_training_samples)
    agg_weights = [
        x/total_seen_training_samples for x in seen_training_samples]

    return agg_weights

# Different schemes for choosing number of guesses

def get_num_guesses(n_guesses, num_clients, budgets, lower_bound, upper_bound):
    max_u = max(budgets)
    if(n_guesses == 'max'):
        num_gradient_guesses = [max_u - c for c in budgets]
    elif(n_guesses == 'best'):
        mid_u = int((lower_bound + upper_bound) / 2.0)
        gap = upper_bound - mid_u
        num_gradient_guesses = [abs(int((gap - abs(mid_u - b))/gap * 10)) for b in budgets]
    elif(n_guesses[0] == 'g'):
        percentage_g = float(n_guesses[1:])/100.0
        num_gradient_guesses = [int(floor(percentage_g * b)) for b in budgets]
    elif(n_guesses[0] == 'b'):
        boosting_g = int(n_guesses[1:])
        num_gradient_guesses = [
            max_u + boosting_g - c for c in budgets]
    elif(n_guesses[0] == 'f'):
        g = int(n_guesses[1:])
        num_gradient_guesses = [min(c + g, max_u) - c for c in budgets]
    # percentage guessing --> p20 would mean 20% of budget
    elif(n_guesses[0] == 'p'):
        percentage_g = float(n_guesses[1:])/100.0
        num_gradient_guesses = [int(floor(percentage_g * b)) for b in budgets]
        budgets = [budgets[i] - num_gradient_guesses[i] for i in range(len(budgets))]
    # percentage guessing --> x20 would mean reduce 20% of budget
    # to simulate the baseline, guesses set to zero
    elif(n_guesses[0] == 'x'):
        percentage_g = float(n_guesses[1:])/100.0
        num_gradient_guesses = [int(floor(percentage_g * b)) for b in budgets]
        budgets = [budgets[i] - num_gradient_guesses[i] for i in range(len(budgets))]
        num_gradient_guesses = [0]*num_clients
    else:
        num_gradient_guesses = [int(n_guesses)]*num_clients

    return num_gradient_guesses, budgets





def run_fl(
    # Random number generator for reproducible client and budget selection
    rng: np.random.Generator,
    dataset,  # Standardised dataset object
    preprocess,  # For dataset specfic processing - DD
    central_test_dataset,  # Preprocessed central test dataset - DD
    evaluate,  # Function encapsulating keras compiled model for test set evaluation -  DD
    check_stopping_criteria,  # Function to evaluate stopping criteria - DD
    hparams,  # Hyperparameters, currently just contains batch size
    log_dir,  # Directory for logging
    lower_bound,  # lower bound for client budgets
    upper_bound,  # upper bound for client budgets
    total_clients,  # Total number of clients in the dataset
    num_clients=10,  # No of clients selected per round
    fixed_rounds=None,  # If specified, overrides check for stopping criteria
    evaluate_every=3,  # How often to evaluate on test set
    lr_schedule=None,  # f: int -> float ; takes round number as input
    n_guesses=0,  # Is either string 'max' or an int for fixed no. of guesses
    model_weights_file=None):  # File containing model weights for consistent initialisation

    iterative_process = tff.templates.IterativeProcess(
        initialize_fn=initialize_fn,
        next_fn=next_fn
    )

    state = iterative_process.initialize()

    if(model_weights_file is not None):
        print('[Loading weights from file]')
        keras_model = keras_model_fn()
        keras_model.load_weights(model_weights_file)
        state = [v.numpy() for v in keras_model.trainable_weights]

    print('[Beginning training]')

    best_accuracy = None
    best_accuracy_round_num = None
    round_num = 1

    train_accuracies = []
    train_losses = []
    train_index = []

    test_accuracies = []
    test_losses = []
    test_index = []


    all_budgets = []
    all_guesses = []

    with tf.summary.create_file_writer(log_dir).as_default():
        while(True):
            print("[Round {}]".format(round_num))

            # Evaluate before beginning any training
            if(round_num == 1):
                print('[Evaluating on test data...]')
                test_loss, test_accuracy = evaluate(
                    state, central_test_dataset)

                best_accuracy = test_accuracy
                best_accuracy_round_num = round_num

                tf.summary.scalar('test_loss', test_loss, step=0)
                tf.summary.scalar('test_accuracy', best_accuracy, step=0)

                print(
                    '[Test loss - {} Test accuracy - {}]'.format(test_loss, test_accuracy))
                test_index.append(0)
                test_accuracies.append(best_accuracy)
                test_losses.append(test_loss)

            client_indexes = rng.integers(
                low=0, high=total_clients, size=num_clients)

            client_ids = []
            client_num_samples = []
            for client_index in client_indexes:
                client_ids.append(dataset.client_ids[client_index])
                client_num_samples.append(dataset.num_samples[client_index])

            budgets = rng.integers(
                low=lower_bound, high=upper_bound+1, size=num_clients)

            print("Capacities [", *budgets, "]")
            print("Num samples [", *client_num_samples, "]")

            # ------- Compute/fetch all things necessary for training
            lr_to_clients = [lr_schedule(round_num)]*num_clients
            num_gradient_guesses, budgets = get_num_guesses(
                n_guesses, num_clients, budgets, lower_bound, upper_bound)
            client_agg_weights = get_client_agg_weights(
                budgets, client_num_samples, hparams['batch_size'])

            federated_train_data = make_federated_data(
                dataset, preprocess, client_ids, client_num_samples, budgets, hparams['batch_size'], round_num)

            all_budgets.append(budgets)
            all_guesses.append(num_gradient_guesses)

            total_work = [budgets[i] + num_gradient_guesses[i]
                          for i in range(len(budgets))]
            print("Final total work [", *total_work, "]")

            # ---------- Train one round
            state, metrics = iterative_process.next(
                state,
                federated_train_data,
                lr_to_clients,
                num_gradient_guesses,
                client_agg_weights
                )

            for name, value in metrics.items():
                tf.summary.scalar('train_' + name, value, step=round_num)
                if('loss' in name):
                    print('[Train loss', value, ']')
                    train_losses.append(value)
                else:
                    print('[Train accuracy', value, ']')
                    train_accuracies.append(value)
            train_index.append(round_num)

            if((round_num+2) % evaluate_every == 0):
                print('[Evaluating on test data...]')
                test_loss, test_accuracy = evaluate(
                    state, central_test_dataset)

                if(best_accuracy is None or test_accuracy > best_accuracy):
                    best_accuracy = test_accuracy
                    best_accuracy_round_num = round_num

                tf.summary.scalar('test_loss', test_loss, step=round_num)
                tf.summary.scalar(
                    'test_accuracy', best_accuracy, step=round_num)

                print(
                    '[Test loss - {} Test accuracy - {}]'.format(test_loss, test_accuracy))
                test_index.append(round_num)
                test_accuracies.append(best_accuracy)
                test_losses.append(test_loss)
                if(fixed_rounds is not None and round_num >= fixed_rounds):
                    return best_accuracy, best_accuracy_round_num, round_num, state, all_budgets, all_guesses, (train_accuracies, train_losses, train_index), (test_accuracies, test_losses, test_index)
                elif(fixed_rounds is None and check_stopping_criteria(test_losses[-10:], test_accuracies[-10:], best_accuracy, round_num)):
                    return best_accuracy, best_accuracy_round_num, round_num, state, all_budgets, all_guesses, (train_accuracies, train_losses, train_index), (test_accuracies, test_losses, test_index)

            round_num = round_num + 1
