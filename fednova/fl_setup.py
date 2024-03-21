import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
import os

nest_asyncio.apply()
np.random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#------------------------------------------------------------------------------  
# SERVER_STATE = {model weights}
@tff.tf_computation
def server_init():
    model = model_fn()
    return model.trainable_variables

# {model weights}@SERVER
@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)

#------------------------------------------------------------------------------
# Defining type signatures - 1
model_weights_type = server_init.type_signature.result
dummy_model = model_fn()
tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
federated_server_state_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

#------------------------------------------------------------------------------
@tf.function
def client_update(model, dataset, server_weights, lr, num_guesses, accumulator):
    
    """Performs training (using the server model weights) on the client's dataset."""
    # Initialize the client model with the current server weights.
    client_weights = model.trainable_variables
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    d_size = dataset.cardinality()
    for batch in dataset:
        with tf.GradientTape(persistent=True) as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights)

        # Apply the gradient using a client optimizer.
        updated_accumulator = tf.nest.map_structure(lambda a, g: 0.9*a - lr*g, accumulator, grads)
        updated_weights = tf.nest.map_structure(lambda w, a: w+a, client_weights, updated_accumulator)
        
        tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, updated_weights)
        tf.nest.map_structure(lambda x, y: x.assign(y), accumulator, updated_accumulator)

    scaling_factor = (1.0-tf.pow(0.9, tf.cast(num_guesses, tf.float32)))*9.0
    client_weights = tf.nest.map_structure(lambda x, y: tf.add(x, y * scaling_factor),
                        client_weights, accumulator)
    
    # delta_i = x_i^(t, tau_i) - x_i^(t, 0)
    model_delta = tf.nest.map_structure(tf.subtract, client_weights, server_weights)
    
    out_data = model.report_local_outputs()
    
    return model_delta, out_data, out_data['loss'][1]

@tff.tf_computation(
    tf_dataset_type, 
    model_weights_type, 
    tf.float32,  # learning rate
    tf.int32)    # number of guesses
def client_update_fn(tf_dataset, server_weights_at_client, learning_rate, num_guesses):
    
    model = model_fn()
    
    accumulator = tf.nest.map_structure(lambda l: tf.Variable(tf.zeros(l.shape, l.dtype)), server_weights_at_client)
    
    client_weights, out_data, n = client_update(model, tf_dataset, 
        server_weights_at_client, learning_rate, num_guesses, accumulator)

    return client_weights, out_data, n

#------------------------------------------------------------------------------
@tf.function
def server_update(current_server_state, aggregated_client_model_deltas, server_scaling_factor):
    # Update server model as scaled aggregated model deltas
    return tf.nest.map_structure(lambda x, y: x + (y * server_scaling_factor),
                        current_server_state, aggregated_client_model_deltas)
    
@tff.tf_computation(
    model_weights_type, # x^(t, 0)
    model_weights_type, # aggregated client deltas
    tf.float32) # tau_eff
def server_update_fn(current_server_state, aggregated_client_model_deltas, server_scaling_factor):
    # Computes x^(t+1, 0) = x^(t, 0) + (tau_eff x sum p_i x d_i x -eta)
    return server_update(current_server_state, aggregated_client_model_deltas, server_scaling_factor)

#------------------------------------------------------------------------------
# Defining type signatures - 2
# For simulating varying epochs across clients
client_learning_rates_type = tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=False)
client_num_guesses_type = tff.FederatedType(tf.int32, tff.CLIENTS, all_equal=False)
client_coeff_norm_type = tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=False)
client_agg_weights_type = tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=False)
server_scaling_factor_type = tff.FederatedType(tf.float32, tff.SERVER)

#------------------------------------------------------------------------------    
@tff.tf_computation(client_update_fn.type_signature.result)
def extract_weights(tp_wts_mts):
    return tp_wts_mts[0], tp_wts_mts[2] 

@tff.tf_computation(client_update_fn.type_signature.result)
def extract_only_weights(tp_wts_mts):
  return tp_wts_mts[0]

@tff.tf_computation(client_update_fn.type_signature.result)
def extract_training_metrics(tp_wts_mts):
    return tp_wts_mts[1]

# Receives a dictionary {metric: [sum_all_samples, total_samples]}
# Return dictionary of means for every metric
@tff.tf_computation(client_update_fn.type_signature.result[1])
def get_mean(metric_dict):
    d = {}
    for k,v in metric_dict.items():
        d[k] = v[0]/v[1] 
    return d

@tff.tf_computation(model_weights_type, tf.float32)
def fed_avg(model_weights, n_k):
    return tf.nest.map_structure(lambda x: x * n_k,
                        model_weights), n_k

@tff.tf_computation(model_weights_type, tf.float32)
def do_div(summed_weights, c):
    return tf.nest.map_structure(lambda x: x / c,
                        summed_weights)

#------------------------------------------------------------------------------
@tff.federated_computation(
    federated_server_state_type, 
    federated_dataset_type, 
    client_learning_rates_type,
    client_num_guesses_type, # List of g_i
    client_coeff_norm_type,  # List of ||a_i||_1
    client_agg_weights_type, # List of p_i = n_i/n
    server_scaling_factor_type) # tau_eff
def next_fn(
    server_state, 
    federated_dataset, 
    client_learning_rates, 
    client_num_guesses, 
    client_coeff_norms,
    client_agg_weights,
    server_scaling_factor):

    # Broadcast the server weights to the clients.
    server_weights_at_clients = tff.federated_broadcast(server_state)

    # Each client computes their updated weights
    # and returns delta_i
    client_weights_and_metrics = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_clients, client_learning_rates, client_num_guesses))
    
    client_model_deltas = tff.federated_map(extract_only_weights, client_weights_and_metrics)
    client_metrics = tff.federated_map(extract_training_metrics, client_weights_and_metrics)
    
    # delta_i / ||a_i||_1  = -eta x Ga_i / (||a_i||_1)
    normalized_client_weights = tff.federated_map(do_div, (client_model_deltas, client_coeff_norms))
    
    # sum p_i x d_i x -eta = sum p_i x delta_i / ||a_i||_1
    aggregated_client_model_deltas = tff.federated_mean(normalized_client_weights, client_agg_weights)
    
    # compute mean of training metrics
    client_metrics_summed = tff.federated_sum(client_metrics)
    mean_metrics = tff.federated_map(get_mean, client_metrics_summed)

    # x^(t+1, 0) = x^(t, 0) + (tau_eff x sum p_i x d_i x -eta)
    server_state = tff.federated_map(server_update_fn, (server_state, aggregated_client_model_deltas, server_scaling_factor))

    return server_state, mean_metrics