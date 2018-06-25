from cluster import WorkerPool
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

onehot_words, word_pos, sentence_lens_nchars, sentence_lens_nwords, vocabulary_size, max_char_len = read(
    file='train.h5', train=True)
onehot_words_val, word_pos_val, sentence_lens_nchars_val, sentence_lens_nwords_val, _, _ = read(file='train.h5',
                                                                                                train=False)

max_char_len = 494
batch_size = 52
hidden_size = 1024
decoder_dim = 1024
vocabulary = ["<SOS>"] + ["a"] + ["b"] + ["c"] + ["d"] + ["e"] + ["f"] + \
             ["g"] + ["h"] + ["i"] + ["j"] + ["k"] + ["l"] + ["m"] + ["n"] + ["o"] + \
             ["p"] + ["q"] + ["r"] + ["s"] + ["t"] + ["u"] + ["v"] + ["w"] + \
             ["x"] + ["y"] + ["z"] + ["<EOW>"] + ["<EOS>"] + [">"] + ["-"] + ["."] + ["'"] + ["0"] + ["1"] + ["2"] + [
                 "3"] + \
             ["4"] + ["5"] + ["6"] + ["7"] + ["8"] + ["9"] + ["&"] + ["<"] + ["$"] + ["#"] + ["/"] + [","] + ["|"] + \
             ["@"] + ["%"] + ["^"] + ["\\"] + ["*"] + ["("] + [")"] + ["{"] + ["}"] + [":"] + [";"]

vocabulary_size = len(vocabulary)
# token2index = {token:index for index,token in enumerate(vocabulary)}
index2token = {index: token for index, token in enumerate(vocabulary)}

train_dict = {'batch_size': 52, 'hidden_size': hidden_size, 'decoder_dim': decoder_dim, 'max_char_len': max_char_len,
              'onehot_words': onehot_words, 'word_pos': word_pos, 'sentence_lens_nchars': sentence_lens_nchars,
              'vocabulary_size': vocabulary_size, 'sentence_lens_nwords': sentence_lens_nwords,
              'onehot_words': onehot_words_val, 'word_pos': word_pos_val,
              'sentence_lens_nchars': sentence_lens_nchars_val, 'sentence_lens_nwords_val': sentence_lens_nwords_val}

network_dict = {'max_char_len': max_char_len, 'batch_size': batch_size, 'hidden_size': hidden_size}

log("start")
train(n_epochs=1, network_dict=network_dict, index2token=index2token, **train_dict)
log("end")

if __name__ == '__main__':
    exp_name = 'vae_model'

    pool = WorkerPool(1, worker_types='gpu_gtx_1080_ti')

    pool.attach_files(['train_vaeLM.py', 'decoder.py', 'encoder.py', 'cluster_print.py', 'pre.py', h5py])

    pool(experiment, [exp_name])
