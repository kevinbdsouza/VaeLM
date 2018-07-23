import h5py
from sys import argv


def experiment(exp_name, logdir, prep_file_dir):
    from decoder import Decoder
    import encoder
    import tensorflow as tf
    import numpy as np
    from functools import reduce
    import tensorflow as tf
    import numpy as np
    from tensorflow.python.ops.rnn import _transpose_batch_time
    import collections
    import os
    import argparse
    import datetime as dt
    from collections import Counter
    from random import random
    from nltk import word_tokenize
    from train_vaeLM import train, prep_perm_matrix, permute_encoder_output
    from pre import read
    onehot_words, word_pos, sentence_lens_nchars, sentence_lens_nwords, vocabulary_size, max_char_len = read(
        file_name=prep_file_dir + 'train.h5', train=True)
    onehot_words_val, word_pos_val, sentence_lens_nchars_val, sentence_lens_nwords_val, _, _ = read(
        file_name=prep_file_dir + 'test.h5', train=False)

    max_char_len = 371
    batch_size = 40
    hidden_size = 512
    decoder_dim = 512
    decoder_units_p3 = 512
    vocabulary = ["<SOS>"] + ["a"] + ["b"] + ["c"] + ["d"] + ["e"] + ["f"] + \
                 ["g"] + ["h"] + ["i"] + ["j"] + ["k"] + ["l"] + ["m"] + ["n"] + ["o"] + \
                 ["p"] + ["q"] + ["r"] + ["s"] + ["t"] + ["u"] + ["v"] + ["w"] + \
                 ["x"] + ["y"] + ["z"] + ["<EOW>"] + ["<EOS>"] + [">"] + ["-"] + ["."] + ["'"] + ["0"] + ["1"] + [
                     "2"] + ["3"] + \
                 ["4"] + ["5"] + ["6"] + ["7"] + ["8"] + ["9"] + ["&"] + ["<"] + ["$"] + ["#"] + ["/"] + [","] + ["|"] + \
                 ["@"] + ["%"] + ["^"] + ["\\"] + ["*"] + ["("] + [")"] + ["{"] + ["}"] + [":"] + [";"]

    vocabulary_size = len(vocabulary)
    # token2index = {token:index for index,token in enumerate(vocabulary)}
    index2token = {index: token for index, token in enumerate(vocabulary)}

    train_dict = {'decoder_units_p3': decoder_units_p3, 'batch_size': batch_size, 'hidden_size': hidden_size,
                  'decoder_dim': decoder_dim, 'max_char_len': max_char_len, 'onehot_words': onehot_words,
                  'word_pos': word_pos, 'sentence_lens_nchars': sentence_lens_nchars,
                  'vocabulary_size': vocabulary_size, 'sentence_lens_nwords': sentence_lens_nwords,
                  'onehot_words_val': onehot_words_val, 'word_pos_val': word_pos_val,
                  'sentence_lens_nchars_val': sentence_lens_nchars_val,
                  'sentence_lens_nwords_val': sentence_lens_nwords_val}

    network_dict = {'max_char_len': max_char_len, 'batch_size': batch_size, 'hidden_size': hidden_size}

    train(log_dir=log_dir, n_epochs=500, network_dict=network_dict, index2token=index2token, **train_dict)


if __name__ == '__main__':
    log_dir = argv[-1]
    prep_file_dir = argv[-2]
    exp_name = 'vae_model'

    experiment(exp_name, logdir=log_dir, prep_file_dir=prep_file_dir)
