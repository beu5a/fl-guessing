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

def read_dir(data_dir):
    clients = []
    num_samples = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            client_data = json.load(inf)
        clients.extend(client_data['users'])
        num_samples.extend(client_data['num_samples'])
        data.update(client_data['user_data'])

    return clients, num_samples, data
    
class FemnistData:
    def __init__(self, train_dir, test_dir):
        self.client_ids, self.num_samples, self.train_data = read_dir(train_dir)
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

def femnist_batch_format_fn(x, y):
    # Modify shape (28, 28) to (28, 28, 1)
    return (tf.reshape(x, [-1, 28, 28, 1]),
        tf.expand_dims(y, axis=-1))

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

def make_federated_data(dataset, preprocess_fn, client_ids, client_num_samples, client_capacities, batch_size, round_num):
    return [
      preprocess_fn(dataset.create_dataset_for_client(client_ids[i]), batch_size, client_capacities[i], client_num_samples[i], round_num)
      for i in range(len((client_ids)))
    ]

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