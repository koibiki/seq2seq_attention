import os

import tensorflow as tf

from lang_dict.lang_dict import LanguageDict
from utils.seq_utils import preprocess_sentence, max_length


class DataProvider:

    def __init__(self, batch_size):
        # Download the file
        path_to_zip = tf.keras.utils.get_file(
            'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',
            extract=True)

        path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

        self.batch_size = batch_size

        num_examples = 300000
        input_tensor, target_tensor, inp_lang, targ_lang, input_seq_len, target_seq_len = \
            self.load_dataset(path_to_file, num_examples)

        self.inp_lang = inp_lang
        self.targ_lang = targ_lang

        train_size = int(0.95 * len(input_tensor))
        input_tensor_train = input_tensor[:train_size]
        input_tensor_val = input_tensor[train_size:]
        target_tensor_train = target_tensor[:train_size]
        target_tensor_val = target_tensor[train_size:]
        input_seq_len_train = input_seq_len[:train_size]
        input_seq_len_val = input_seq_len[train_size:]
        target_seq_len_train = target_seq_len[:train_size]
        target_seq_len_val = target_seq_len[train_size:]

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train, input_seq_len_train, target_seq_len_train)) \
            .shuffle(3).repeat()
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_val, target_tensor_val, input_seq_len_val, target_seq_len_val)).repeat()
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

        self.train_dataset = train_dataset

        self.val_dataset = val_dataset

        self.train_n_batch = len(input_tensor_train) // batch_size

        self.val_n_batch = len(input_tensor_val) // batch_size

    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    def create_dataset(self, path, num_examples):
        lines = open(path, encoding='UTF-8').read().strip().split('\n')

        word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

        return word_pairs

    def load_dataset(self, path, num_examples):
        # creating cleaned input, output pairs
        pairs = self.create_dataset(path, num_examples)

        # index language using the class defined above
        inp_lang = LanguageDict(sp for en, sp in pairs)
        targ_lang = LanguageDict(en for en, sp in pairs)

        # Vectorize the input and target languages
        # Spanish sentences
        input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

        # English sentences
        target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]

        # Calculate max_length of input and output tensor
        # Here, we'll set those to the longest sentence in the dataset
        max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

        input_seq_len = [len(sp.split(' ')) for en, sp in pairs]
        target_seq_len = [len(en.split(' ')) for en, sp in pairs]

        # Padding the input and output tensor to the maximum length
        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                     maxlen=max_length_inp,
                                                                     padding='post',
                                                                     value=inp_lang.word2idx["<PAD>"])

        target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                      maxlen=max_length_tar,
                                                                      padding='post',
                                                                      value=targ_lang.word2idx["<PAD>"])

        return input_tensor, target_tensor, inp_lang, targ_lang, input_seq_len, target_seq_len
