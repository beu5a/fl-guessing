import tensorflow as tf
import os
import json
import numpy as np
from collections import defaultdict
from math import floor, ceil
from PIL import Image
import re
import pickle
import collections
import tensorflow_datasets as tfds
from typing import Tuple
import tensorflow_federated as tff

#------------------------------------------------------------------------------
def read_dir(data_dir):
    clients = []
    num_samples = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            client_data = json.load(inf)
        clients.extend(client_data['users'])
        num_samples.extend(client_data['num_samples'])
        data.update(client_data['user_data'])

    return clients, num_samples, data


class FemnistData:
    def __init__(self, train_dir, test_dir):
        self.client_ids, self.num_samples, self.train_data = read_dir(
            train_dir)
        _, _, self.test_data = read_dir(test_dir)

    def get_client_ids(self):
        return self.client_ids

    def create_dataset_for_client(self, client_id):
        client_data = self.train_data[client_id]
        return tf.data.Dataset.from_tensor_slices((client_data['x'], client_data['y']))

    def create_train_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.train_data.values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

    def create_test_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.test_data.values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

#------------------------------------------------------------------------------
class CIFAR10Data:
    def __init__(self):
        self.CIFAR_SHAPE = (32, 32, 3)
        self.TOTAL_FEATURE_SIZE = 32 * 32 * 3
        self.NUM_CLASSES = 10
        self.TRAIN_EXAMPLES = 50000
        self.TEST_EXAMPLES = 10000
        self.NUM_CLIENTS = 16
        self.DIRICHLET_PARAM = 0.1
        # Number of training examples per class: 50,000 / 10.
        self.TRAIN_EXAMPLES_PER_LABEL = 5000
        # Number of test examples per class: 10,000 / 10.
        self.TEST_EXAMPLES_PER_LABEL = 1000
        self.train_data, self.test_data,  self.client_ids = self.load_cifar10_federated(self.DIRICHLET_PARAM, self.NUM_CLIENTS)
        self.num_samples = [
            int(len(self.train_data[str(id)]['label'])) for id in self.client_ids]

    # This function has been borrowred from 
    # https://github.com/google-research/federated/blob/master/utils/datasets/cifar10_dataset.py
    
    def load_cifar10_federated(
        self,
        dirichlet_parameter: float = 0.1,
        num_clients: int = 16,
    ) :
        """Construct a federated dataset from the centralized CIFAR-10.
        Sampling based on Dirichlet distribution over categories, following the paper
        Measuring the Effects of Non-Identical Data Distribution for
        Federated Visual Classification (https://arxiv.org/abs/1909.06335).
        """
        train_images, train_labels = tfds.as_numpy(
            tfds.load(
                name='cifar10',
                split='train',
                batch_size=-1,
                as_supervised=True,
            ))
        test_images, test_labels = tfds.as_numpy(
            tfds.load(
                name='cifar10',
                split='test',
                batch_size=-1,
                as_supervised=True,
            ))


        train_images = train_images 
        test_images = test_images


        train_clients = collections.OrderedDict()
        test_clients = collections.OrderedDict()

        train_multinomial_vals = []
        test_multinomial_vals = []

        # Each client has a multinomial distribution over classes drawn from a
        # Dirichlet.


        for i in range(num_clients):
            proportion = np.random.dirichlet(dirichlet_parameter *
                                            np.ones(self.NUM_CLASSES,))
            train_multinomial_vals.append(proportion)
            test_multinomial_vals.append(proportion)

        train_multinomial_vals = np.array(train_multinomial_vals)
        test_multinomial_vals = np.array(test_multinomial_vals)

        train_example_indices = []
        test_indices = []
        for k in range(self.NUM_CLASSES):
            train_label_k = np.where(train_labels == k)[0]
            np.random.shuffle(train_label_k)
            train_example_indices.append(train_label_k)
            test_label_k = np.where(test_labels == k)[0]
            np.random.shuffle(test_label_k)
            test_indices.append(test_label_k)

        train_example_indices = np.array(train_example_indices)
        test_indices = np.array(test_indices)

        train_client_samples = [[] for _ in range(num_clients)]
        test_client_samples = [[] for _ in range(num_clients)]
        train_count = np.zeros(self.NUM_CLASSES).astype(int)
        test_count = np.zeros(self.NUM_CLASSES).astype(int)

        train_examples_per_client = int(self.TRAIN_EXAMPLES / num_clients)
        test_examples_per_client = int(self.TEST_EXAMPLES / num_clients)



        for k in range(num_clients):

            for i in range(train_examples_per_client):
                sampled_label = np.argwhere(
                    np.random.multinomial(1, train_multinomial_vals[k, :]) == 1)[0][0]
                train_client_samples[k].append(
                    train_example_indices[sampled_label, train_count[sampled_label]])
                train_count[sampled_label] += 1
                if train_count[sampled_label] == self.TRAIN_EXAMPLES_PER_LABEL:
                    train_multinomial_vals[:, sampled_label] = 0
                    if (np.all(train_multinomial_vals.sum(axis=1))):
                        train_multinomial_vals = (
                            train_multinomial_vals /
                            train_multinomial_vals.sum(axis=1)[:, None])

            for i in range(test_examples_per_client):
                sampled_label = np.argwhere(
                    np.random.multinomial(1, test_multinomial_vals[k, :]) == 1)[0][0]
                test_client_samples[k].append(test_indices[sampled_label,
                                                            test_count[sampled_label]])
                test_count[sampled_label] += 1
                if test_count[sampled_label] == self.TEST_EXAMPLES_PER_LABEL:
                    test_multinomial_vals[:, sampled_label] = 0
                    if (np.all(test_multinomial_vals.sum(axis=1))):
                        test_multinomial_vals = (
                            test_multinomial_vals / test_multinomial_vals.sum(axis=1)[:, None])


        client_ids = []
        for i in range(num_clients):
            client_name = str(i)
            client_ids.append(client_name)
            x_train = train_images[np.array(train_client_samples[i])]
            y_train = train_labels[np.array(
                train_client_samples[i])].astype('int64').squeeze()
            train_data = collections.OrderedDict(
                (('image', x_train), ('label', y_train)))
            train_clients[client_name] = train_data

            x_test = test_images[np.array(test_client_samples[i])]
            y_test = test_labels[np.array(
                test_client_samples[i])].astype('int64').squeeze()
            test_data = collections.OrderedDict((('image', x_test), ('label', y_test)))
            test_clients[client_name] = test_data


        return train_clients, test_clients, client_ids


    def create_dataset_for_client(self, client_id):
        client_data = self.train_data[str(client_id)]
        return tf.data.Dataset.from_tensor_slices((client_data['image'], client_data['label']))
    
    def create_train_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.train_data.values():
            for x in data['image']:
                xs.append(x)
            for y in data['label']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

    def create_test_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.test_data.values():
            for x in data['image']:
                xs.append(x)
            for y in data['label']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

#------------------------------------------------------------------------------
# A fixed vocabularly of ASCII chars that occur in the works of Shakespeare and Dickens:
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
vocab_len = len(vocab)
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Construct a lookup table to map string chars to indexes,
# using the vocab loaded above:
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(len(vocab))),
                                       dtype=tf.int64)),
    default_value=0)

class ShakespeareData:
    def __init__(self, train_dir, test_dir):
        train_file = os.path.join(train_dir, os.listdir(train_dir)[0])
        with open(train_file) as f:
            self.train_json = json.load(f)
        
        test_file = os.path.join(test_dir, os.listdir(test_dir)[0])
        with open(test_file) as f:
            self.test_json = json.load(f)
    
        self.client_ids = self.train_json['users']
        self.num_samples = self.train_json['num_samples']

    def get_client_ids(self):
        return self.client_ids
    
    def create_dataset_for_client(self, client_id):
        client_data = self.train_json['user_data'][client_id]
        return tf.data.Dataset.from_tensor_slices((client_data['x'], client_data['y']))
    
    def create_train_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.train_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

    def create_test_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.test_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

#------------------------------------------------------------------------------
class CelebaData:
    def __init__(self, train_dir, test_dir, images_dir):
        self.IMAGES_DIR = images_dir
        self.IMAGE_SIZE = 84

        train_file = os.path.join(train_dir, os.listdir(train_dir)[0])
        with open(train_file) as f:
            self.train_json = json.load(f)
        
        test_file = os.path.join(test_dir, os.listdir(test_dir)[0])
        with open(test_file) as f:
            self.test_json = json.load(f)
    
        self.client_ids = self.train_json['users']
        self.num_samples = self.train_json['num_samples']

    def get_client_ids(self):
        return self.client_ids
    
    def create_dataset_for_client(self, client_id):
        client_data = self.train_json['user_data'][client_id]
        return tf.data.Dataset.from_tensor_slices((self.process_x(client_data['x']), self.process_y(client_data['y'])))
    
    def create_train_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.train_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((self.process_x(xs), self.process_y(ys)))

    def create_test_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.test_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((self.process_x(xs), self.process_y(ys)))
    
    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return raw_y_batch

    def _load_image(self, img_name):
        img = Image.open(os.path.join(self.IMAGES_DIR, img_name[:-4] + '.png'))
        img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE)).convert('RGB')
        return np.array(img)

#------------------------------------------------------------------------------
class SyntheticData:
    def __init__(self, train_dir, test_dir):
        train_file = os.path.join(train_dir, os.listdir(train_dir)[0])
        with open(train_file) as f:
            self.train_json = json.load(f)
        
        test_file = os.path.join(test_dir, os.listdir(test_dir)[0])
        with open(test_file) as f:
            self.test_json = json.load(f)
    
        self.client_ids = self.train_json['users']
        self.num_samples = self.train_json['num_samples']

    def get_client_ids(self):
        return self.client_ids
    
    def create_dataset_for_client(self, client_id):
        client_data = self.train_json['user_data'][client_id]
        return tf.data.Dataset.from_tensor_slices((client_data['x'], client_data['y']))
    
    def create_train_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.train_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

    def create_test_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.test_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

#------------------------------------------------------------------------------
class Sent140Data:
    def __init__(self, train_dir, test_dir):
        train_file = os.path.join(train_dir, os.listdir(train_dir)[0])
        with open(train_file) as f:
            self.train_json = json.load(f)
        
        test_file = os.path.join(test_dir, os.listdir(test_dir)[0])
        with open(test_file) as f:
            self.test_json = json.load(f)
    
        self.client_ids = self.train_json['users']
        self.num_samples = self.train_json['num_samples']

    def get_client_ids(self):
        return self.client_ids
    
    def create_dataset_for_client(self, client_id):
        client_data = self.train_json['user_data'][client_id]
        return tf.data.Dataset.from_tensor_slices((self.process_x(client_data['x']), self.process_y(client_data['y'])))
    
    def create_train_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.train_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((self.process_x(xs), self.process_y(ys)))

    def create_test_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.test_json['user_data'].values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((self.process_x(xs), self.process_y(ys)))
    
    def process_x(self, raw_x_batch, max_words=25):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [self._line_to_indices(e, word2id, max_words) for e in x_batch]
        x_batch = np.array(x_batch, dtype=np.int32)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [1 if(e=='4') else 0 for e in raw_y_batch]
        y_batch = np.array(y_batch)
        return y_batch

    # Function borrowed from leaf GitHub codebase
    def _line_to_indices(self, line, word2id, max_words=25):
        '''converts given phrase into list of word indices
        
        if the phrase has more than max_words words, returns a list containing
        indices of the first max_words words
        if the phrase has less than max_words words, repeatedly appends integer 
        representing unknown index to returned list until the list's length is 
        max_words
        Args:
            line: string representing phrase/sequence of words
            word2id: dictionary with string words as keys and int indices as values
            max_words: maximum number of word indices in returned list
        Return:
            indl: list of word indices, one index for each word in phrase
        '''
        unk_id = len(word2id)
        line_list = self._split_line(line) # split phrase in words
        indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
        indl += [unk_id]*(max_words-len(indl))
        return indl
    
    # Function borrowed from leaf GitHub codebase
    def _split_line(self, line):
        '''split given line/phrase into list of words
        Args:
            line: string representing phrase to be split
        
        Return:
            list of strings, with each string representing a word
        '''
        return re.findall(r"[\w']+|[.,!?;]", line)

#------------------------------------------------------------------------------
class RedditData:
    def __init__(self, train_dir, test_dir):
        train_file = os.path.join(train_dir, os.listdir(train_dir)[0])
        with open(train_file) as f:
            self.train_json = json.load(f)
        
        test_file = os.path.join(test_dir, os.listdir(test_dir)[0])
        with open(test_file) as f:
            self.test_json = json.load(f)
    
        self.client_ids = self.train_json['users']
        self.num_samples = self.train_json['num_samples']

        VOCABULARY_PATH = os.path.join(train_dir, '../../vocab/reddit_vocab.pck')
        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = self._load_vocab(VOCABULARY_PATH)

    def get_client_ids(self):
        return self.client_ids
    
    def create_dataset_for_client(self, client_id):
        client_data = self.train_json['user_data'][client_id]
        return self.prepare_data(client_data)
    
    def create_train_dataset_for_all_clients(self):
        ds = None
        for data in self.train_json['user_data'].values():
            if(ds is None):
                ds = self.prepare_data(data)
            else:
                ds = ds.concatenate(self.prepare_data(data))
        return ds

    def create_test_dataset_for_all_clients(self):
        ds = None
        for data in self.test_json['user_data'].values():
            if(ds is None):
                ds = self.prepare_data(data)
            else:
                ds = ds.concatenate(self.prepare_data(data))
        return ds
    
    def prepare_data(self, data):
        data_x = data['x']
        data_y = data['y']

        perm = np.random.permutation(len(data['x']))
        data_x = [data_x[i] for i in perm]
        data_y = [data_y[i] for i in perm]

        # flatten lists
        def flatten_lists(data_x_by_comment, data_y_by_comment):
            data_x_by_seq, data_y_by_seq = [], []
            for c, l in zip(data_x_by_comment, data_y_by_comment):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l['target_tokens'])

            return data_x_by_seq, data_y_by_seq
        
        data_x, data_y = flatten_lists(data_x, data_y)

        data_x_processed = self.process_x(data_x)
        data_y_processed = self.process_y(data_y)

        filtered_x, filtered_y = [], []
        for i in range(len(data_x_processed)):
            if(np.sum(data_y_processed[i]) != 0):
                filtered_x.append(data_x_processed[i])
                filtered_y.append(data_y_processed[i])

        return tf.data.Dataset.from_tensor_slices((filtered_x, filtered_y))
    
    def _tokens_to_ids(self, raw_batch):
        def tokens_to_word_ids(tokens, word2id):
            return [word2id[word] for word in tokens]

        to_ret = [tokens_to_word_ids(seq, self.vocab) for seq in raw_batch]
        return np.array(to_ret)

    def process_x(self, raw_x_batch):
        tokens = self._tokens_to_ids([s for s in raw_x_batch])
        return tokens
        
    def process_y(self, raw_y_batch):
        tokens = self._tokens_to_ids([s for s in raw_y_batch])
        
        def getNextWord(token_ids):
            n = len(token_ids)
            for i in range(n):
                if(token_ids[n-i-1] != self.pad_symbol):
                    return token_ids[n-i-1]
            return self.pad_symbol
        
        return [getNextWord(t) for t in tokens]

    def _load_vocab(self, VOCABULARY_PATH):
        vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])

        return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

#------------------------------------------------------------------------------
def femnist_batch_format_fn(x, y):
    # Modify shape (28, 28) to (28, 28, 1)
    return (tf.reshape(x, [-1, 28, 28, 1]),
        tf.expand_dims(y, axis=-1))

def cifar10_batch_format_fn(x,y):
    crop_shape =  (32, 32, 3)
    aux = tf.cast(x, tf.float32)
    aux = tf.image.resize_with_crop_or_pad(
                aux, target_height=crop_shape[0],
                target_width=crop_shape[1])
    aux = tf.image.per_image_standardization(aux)
    
    return (aux, tf.expand_dims(y, axis=-1))

def celeba_batch_format_fn(x, y):
    return (x, tf.expand_dims(y, axis=-1))

def synthetic_batch_format_fn(x, y):
    return (x, tf.expand_dims(y, axis=-1))

def sent140_batch_format_fn(x, y):
    return (x, tf.expand_dims(y, axis=-1))

def reddit_batch_format_fn(x, y):
    return (x, tf.expand_dims(y, axis=-1))

def to_ids(x, y):
    chars = tf.strings.bytes_split(x)
    ids = table.lookup(chars)
    labels = table.lookup(y)
    ids = tf.reshape(ids, shape=[80])
    return ids, tf.one_hot(labels, vocab_len)

#------------------------------------------------------------------------------
def make_federated_data(dataset, preprocess_fn, client_ids, client_num_samples, client_capacities, batch_size, round_num):
    return [
      preprocess_fn(dataset.create_dataset_for_client(client_ids[i]), batch_size, client_capacities[i], client_num_samples[i], round_num)
      for i in range(len((client_ids)))
    ]

#------------------------------------------------------------------------------
def preprocess_shakespeare(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).map(to_ids).batch(b).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).map(to_ids).batch(b).prefetch(tf.data.AUTOTUNE).take(u)       

def preprocess_femnist(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).map(femnist_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).map(
            femnist_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)

def preprocess_cifar10(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).map(cifar10_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).map(
            cifar10_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)


def preprocess_celeba(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).map(celeba_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).map(celeba_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)

def preprocess_synthetic(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).map(synthetic_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).map(synthetic_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    
def preprocess_sent140(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).map(sent140_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).map(sent140_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)

def preprocess_reddit(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).map(reddit_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).map(reddit_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)