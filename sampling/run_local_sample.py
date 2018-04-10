import h5py


def experiment(exp_name):
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
    onehot_words, word_pos, sentence_lens_nchars, sentence_lens_nwords, vocabulary_size, max_char_len, word_loc = read(
        file_name='/home/zalperst/PycharmProjects/vae_proj/train.h5', train=True)
    onehot_words_val, word_pos_val, sentence_lens_nchars_val, sentence_lens_nwords_val, _, _, word_loc_val = read(
        file_name='/home/zalperst/PycharmProjects/vae_proj/test.h5', train=False)

    max_char_len = 371
    batch_size = 40
    hidden_size = 256
    decoder_dim = 256
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

    train_dict = {'batch_size': batch_size, 'hidden_size': hidden_size, 'decoder_dim': decoder_dim,
                  'max_char_len': max_char_len, 'onehot_words': onehot_words, 'word_pos': word_pos,
                  'sentence_lens_nchars': sentence_lens_nchars, 'vocabulary_size': vocabulary_size,
                  'sentence_lens_nwords': sentence_lens_nwords, 'onehot_words_val': onehot_words_val,
                  'word_pos_val': word_pos_val, 'sentence_lens_nchars_val': sentence_lens_nchars_val,
                  'sentence_lens_nwords_val': sentence_lens_nwords_val,
                  'word_loc': word_loc, 'word_loc_val': word_loc_val}

    network_dict = {'max_char_len': max_char_len, 'batch_size': batch_size, 'hidden_size': hidden_size}

    print("start")
    train(log_dir='/home/zalperst/PycharmProjects/vae_proj/', n_epochs=1, network_dict=network_dict,
          index2token=index2token, **train_dict)
    print("end")


if __name__ == '__main__':
    exp_name = 'vae_model'

    experiment(exp_name)
