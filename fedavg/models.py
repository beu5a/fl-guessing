import tensorflow_federated as tff
import tensorflow as tf
import collections
import numpy as np

#------------------------------------------------------------------------------
class Block(tf.keras.models.Sequential):
    def __init__(self,n,m):
        super().__init__()
        for i in range(m):
            self.add(tf.keras.layers.Conv2D(filters = n, kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))

class Dense(tf.keras.models.Sequential):
    def __init__(self,n,m=2):
        super().__init__()
        for i in range(m):
            self.add(tf.keras.layers.Dense(units = n, activation = "relu"))

class VGG11(tf.keras.models.Sequential):
    def __init__(self, input_shape, classes, filters = 64):
        super().__init__()
        self.add(tf.keras.layers.InputLayer(input_shape = input_shape))

        # Backbone
        self.add(Block(n = filters * 1, m = 1))
        self.add(Block(n = filters * 2, m = 1))
        self.add(Block(n = filters * 4, m = 2))
        self.add(Block(n = filters * 8, m = 2))
        self.add(Block(n = filters * 8, m = 2))

        # top
        self.add(tf.keras.layers.Flatten())
        self.add(Dense(n = filters * 64))
        self.add(tf.keras.layers.Dense(units = classes,activation = "softmax"))
    
def get_cifar10_cnn():
    CIFAR_SHAPE = (32, 32, 3)
    return VGG11(CIFAR_SHAPE, 10)

def get_cifar10_cnn2():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    #model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))    # num_classes = 10
    return model

#------------------------------------------------------------------------------
def get_stacked_rnn():
    vocab_len = 400000
    seq_len = 25
    embedding_dim = 100
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_len + 1, embedding_dim, 
        input_length=seq_len))
    rnn_cells = [tf.keras.layers.LSTMCell(100) for _ in range(2)]
    stacked_rnn = tf.keras.layers.StackedRNNCells(rnn_cells)
    model.add(tf.keras.layers.RNN(stacked_rnn))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model

#------------------------------------------------------------------------------
def get_synthetic_perceptron():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(60,)),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

#------------------------------------------------------------------------------
# Batch Normalisation is removed since its not clear how to port moving_mean
# and moving_variance from clients to server
# Also facing other technical issues with evaluating keras model with BatchNorm layers
def get_celeba_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(84, 84, 3)),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2, 2, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2, 2, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2, 2, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2, 2, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])

#------------------------------------------------------------------------------
def get_femnist_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (5, 5)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(62, activation='softmax')
    ])
#------------------------------------------------------------------------------
def get_stacked_lstm():
    # 86 is vocab len
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(86, 8, input_length=80))
    rnn_cells = [tf.keras.layers.LSTMCell(256) for _ in range(2)]
    stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
    model.add(tf.keras.layers.RNN(stacked_lstm))
    model.add(tf.keras.layers.Dense(86, activation='softmax'))
    return model

#------------------------------------------------------------------------------
def tff_sent140_model_fn():
    keras_model = get_stacked_rnn()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 25], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int64)),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]) 

def tff_celeba_model_fn():
    keras_model = get_celeba_cnn()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 84, 84, 3], dtype=tf.uint8),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

def tff_cifar10_model_fn():
    keras_model = get_cifar10_cnn2()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

def tff_synthetic_model_fn():
    keras_model = get_synthetic_perceptron()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 60], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

def tff_femnist_model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = get_femnist_cnn()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

def tff_shakespeare_model_fn():
    keras_model = get_stacked_lstm()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 80], dtype=tf.int64),
            tf.TensorSpec(shape=[None, 86], dtype=tf.float32)),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]) 


## ----------- Code below is not ready, is buggy or is still being developed -----------

# To debug - giving negative accuracies
class CustomAccuracyReddit(tf.keras.metrics.Metric):
    def __init__(self, name="custom_accuracy", **kwargs):
        super(CustomAccuracyReddit, self).__init__(name=name, **kwargs)
        self.n_correct_preds = self.add_weight(name="correct_preds", initializer="zeros", dtype=tf.int32)
        self.n_total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.argmax(y_pred, axis=1)
        y_pred = tf.reshape(pred, shape=(-1, 1))
        values = tf.equal(tf.cast(y_true, "int32"), tf.cast(y_pred, "int32"))
        values = tf.cast(values, "int32")

        unk_preds = tf.equal(tf.cast(pred, "int32"), 1)
        unk_preds_as_ints = tf.cast(unk_preds, "int32")

        pad_preds = tf.equal(tf.cast(pred, "int32"), 0)
        pad_preds_as_ints = tf.cast(pad_preds, "int32")

        self.n_correct_preds.assign_add(
            tf.reduce_sum(values) - tf.reduce_sum(unk_preds_as_ints) - tf.reduce_sum(pad_preds_as_ints))
        self.n_total_samples.assign_add(tf.shape(y_true)[0])

    def result(self):
        return tf.math.divide(self.n_correct_preds, self.n_total_samples)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.n_correct_preds.assign(0)
        self.n_total_samples.assign(0)

# Trial code imitating BinaryAccuracy - Works as expected 
class CustomAccuracyReddit1(tf.keras.metrics.Metric):
    def __init__(self, name="custom_accuracy", **kwargs):
        super(CustomAccuracyReddit, self).__init__(name=name, **kwargs)
        self.n_correct_preds = self.add_weight(name="correct_preds", initializer="zeros", dtype=tf.int32)
        self.n_total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "int32")
        self.n_correct_preds.assign_add(tf.reduce_sum(values))
        self.n_total_samples.assign_add(tf.shape(y_true)[0])

    def result(self):
        return tf.math.divide(self.n_correct_preds, self.n_total_samples)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.n_correct_preds.assign(0)
        self.n_total_samples.assign(0)

def get_reddit_rnn():
    vocab_len = 10000
    seq_len = 10
    embedding_dim = 200
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_len, embedding_dim, 
        input_length=seq_len, mask_zero=True))
    rnn_cells = [tf.keras.layers.LSTMCell(256) for _ in range(2)]
    stacked_rnn = tf.keras.layers.StackedRNNCells(rnn_cells)
    model.add(tf.keras.layers.RNN(stacked_rnn))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(vocab_len, activation='softmax'))
    return model

def tff_reddit_model_fn():
    keras_model = get_reddit_rnn()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 