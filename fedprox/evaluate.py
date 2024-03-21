from models import *

def evaluate_femnist(server_model_weights, central_test_dataset):
  keras_model = get_femnist_cnn()
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  keras_model.set_weights(server_model_weights)
  return keras_model.evaluate(central_test_dataset)

def evaluate_shakespeare(server_model_weights, central_test_dataset):
    keras_model = get_stacked_lstm()
    keras_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]  
    )
    keras_model.set_weights(server_model_weights)
    return keras_model.evaluate(central_test_dataset)

def evaluate_celeba(server_model_weights, central_test_dataset):
    keras_model = get_celeba_cnn()
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
    )
    keras_model.set_weights(server_model_weights)
    return keras_model.evaluate(central_test_dataset)

def evaluate_synthetic(server_model_weights, central_test_dataset):
    keras_model = get_synthetic_perceptron()
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
    )
    keras_model.set_weights(server_model_weights)
    return keras_model.evaluate(central_test_dataset)

def evaluate_sent140(server_model_weights, central_test_dataset):
    keras_model = get_stacked_rnn()
    keras_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]  
    )
    keras_model.set_weights(server_model_weights)
    return keras_model.evaluate(central_test_dataset)

def evaluate_reddit(server_model_weights, central_test_dataset):
    keras_model = get_reddit_rnn()
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[CustomAccuracyReddit()]  
    )
    keras_model.set_weights(server_model_weights)
    return keras_model.evaluate(central_test_dataset)