import tensorflow as tf
import numpy as np
import collections
import os
import argparse
import datetime as dt

from collections import Counter
from random import random
from nltk import word_tokenize


def preprocess(mode):
    # load text files
    train_sentences = [line.strip() for line in open("data/english/train.txt").readlines()]
    val_sentences = [line.strip() for line in open("data/english/valid.txt").readlines()]
    test_sentences = [line.strip() for line in open("data/english/test.txt").readlines()]
    train_sentences = [x for x in train_sentences if x]
    val_sentences = [x for x in val_sentences if x]
    test_sentences = [x for x in test_sentences if x]
    max_char_len = 371

    if mode == "train":
        sentences = train_sentences
        pop_list = [4607, 38450, 24213, 27130, 28833, 39006, 38446, 20728, 2066, 11982, 2298, 18158, 4820, 29089, 24112,
                    35834,
                    8573, 30944, 5791, 12130, 10752, 30857, 34030, 458, 35900, 3219, 7860, 10241]
        for pop in pop_list:
            sentences.pop(pop)
    # max_char_len = 494
    elif mode == "val":
        sentences = val_sentences
    # max_char_len = 356
    elif mode == "test":
        sentences = test_sentences
    # max_char_len = 463

    sentences = [["<SOS>"] + word_tokenize(sentence.lower()) + ["<EOS>"] for sentence in sentences]

    # set > as unk
    for ind, sen in enumerate(sentences):
        for i in range(20):
            try:
                sen.remove("<")
                sen.remove("unk")
            except:
                pass

    # define vocab
    vocabulary = ["<SOS>"] + ["a"] + ["b"] + ["c"] + ["d"] + ["e"] + ["f"] + \
                 ["g"] + ["h"] + ["i"] + ["j"] + ["k"] + ["l"] + ["m"] + ["n"] + ["o"] + \
                 ["p"] + ["q"] + ["r"] + ["s"] + ["t"] + ["u"] + ["v"] + ["w"] + \
                 ["x"] + ["y"] + ["z"] + ["<EOW>"] + ["<EOS>"] + [">"] + ["-"] + ["."] + ["'"] + ["0"] + ["1"] + [
                     "2"] + ["3"] + \
                 ["4"] + ["5"] + ["6"] + ["7"] + ["8"] + ["9"] + ["&"] + ["<"] + ["$"] + ["#"] + ["/"] + [","] + ["|"] + \
                 ["@"] + ["%"] + ["^"] + ["\\"] + ["*"] + ["("] + [")"] + ["{"] + ["}"] + [":"] + [";"]

    vocabulary_size = len(vocabulary)
    token2index = {token: index for index, token in enumerate(vocabulary)}
    index2token = {index: token for index, token in enumerate(vocabulary)}
    one_hot_embeddings = np.eye(vocabulary_size)

    # find max word length
    max_word_length = 0
    maxid = 0
    for i in range(len(sentences)):
        l = len(sentences[i])
        if l > max_word_length:
            maxid = i
            max_word_length = l

    return sentences, vocabulary_size, max_word_length, one_hot_embeddings, token2index, max_char_len, index2token


# produce character embeddings
def embed_producer(sentences, vocabulary_size, max_word_length, one_hot_embeddings, token2index, max_char_len):
    s_tensor = np.empty((len(sentences), max_char_len, vocabulary_size))
    word_loc_all = np.zeros((len(sentences), max_word_length))
    eow_loc_all = np.zeros((len(sentences), max_char_len))
    sen_lens = []
    num_words = []
    for i in range(len(sentences)):
        s = sentences[i]
        embed = np.zeros((max_char_len, vocabulary_size))
        word_loc = np.zeros(max_word_length)
        eow_loc = np.zeros(max_char_len)
        prev = 0
        count = 0
        # print(i)
        for k in range(len(s)):
            w = s[k]
            # print(w)
            for id, token in enumerate(w):

                if (w == "<EOS>") | (w == "<SOS>") | (w == ">"):
                    break
                else:
                    # print(prev + id)
                    # print(token)
                    count += 1
                    embed[prev + id, :] = np.squeeze(one_hot_embeddings[token2index.get(token)])

            if (w == "<EOS>") | (w == "<SOS>"):
                word_loc[k] = id + 1
                # print(prev)
                embed[prev, :] = one_hot_embeddings[token2index.get(w)]
                count += 1
                eow_loc[count] = 1
                prev = prev + id + 1

            elif (w == ">"):
                word_loc[k] = id + 1
                count += 1
                embed[prev, :] = one_hot_embeddings[token2index.get(w)]
                prev = prev + id + 1
                embed[prev, :] = one_hot_embeddings[token2index.get("<EOW>")]
                count += 1
                eow_loc[count] = 1
                prev = prev + 1

            else:
                prev = prev + id + 1
                word_loc[k] = id + 1
                # print(prev)
                embed[prev, :] = one_hot_embeddings[token2index.get("<EOW>")]
                count += 1
                eow_loc[count] = 1
                prev = prev + 1

        s_tensor[i, :, :] = embed
        eow_loc_all[i, :] = eow_loc
        n_w = int(np.sum(eow_loc_all[i]))

        num_words.append(2 * n_w - 1)
        sen_lens.append(count + 1)

        # to get word end locations to retrieve hidden states later
        word_loc_all[i, 0] = word_loc[0]
        for j in range(1, len(s)):
            word_loc_all[i, j] = word_loc_all[i, j - 1] + word_loc[j]

    return s_tensor, eow_loc_all, sen_lens, num_words


def run_preprocess(mode):
    # preprocess the data
    sentences, vocabulary_size, max_word_length, one_hot_embeddings, token2index, max_char_len, index2token = preprocess(
        mode)
    # produce embeddings
    data, eow_loc_all, sen_lens, num_words = embed_producer(sentences, vocabulary_size, max_word_length,
                                                            one_hot_embeddings, token2index, max_char_len)

    return data, eow_loc_all, sen_lens, num_words, vocabulary_size, index2token, max_char_len


def get_output_sentences(index2token, indices):
    # indices of size (_,maxChar)
    space = ""
    sentences_all = []
    for sample in range(len(indices)):
        sentence = []
        sen = indices[sample]
        for char in range(len(sen)):
            if (index2token.get(sen[char]) == "<SOS>"):
                sentence.append("")
            elif (index2token.get(sen[char]) == "<EOS>"):
                break
            elif (index2token.get(sen[char]) == "<EOW>"):
                sentence.append(" ")
            else:
                sentence.append(index2token.get(sen[char]))

        sentences_all.append(space.join(sentence))

    return sentences_all


def eow_pos(word_pos, sample_pos, batch_size, max_char_len):
    word_locations = np.zeros([batch_size], dtype=np.float32)

    for btch in range(batch_size):
        count = 0
        samp = word_pos[btch, :]
        for tim in range(max_char_len):
            count = count + samp[tim]
            if (count == sample_pos):
                word_locations[btch] = tim
                break

    return word_locations


class Encoder:
    def __init__(self, **kwargs):
        # self.data =kwargs['data']
        # self.sentences =kwargs['sentences']
        # self.vocabulary_size = kwargs['vocabulary_size']
        # self.max_word_length = kwargs['max_word_length']
        self.max_char_len = kwargs['max_char_len']
        self.batch_size = kwargs['batch_size']
        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']

    # our [494, 52, 61] tensor becomes [[52, 61], [52, 61], ...]
    def run_encoder(self, inputs, word_pos, word_locations, reuse):

        inputs_t = tf.transpose(inputs, perm=[1, 0, 2])
        _inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.max_char_len, name='char_array')
        _inputs_ta = _inputs_ta.unstack(inputs_t)

        cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
        output_ta = tf.TensorArray(size=self.max_char_len, dtype=tf.float32, name='word_array')
        mean_ta = tf.TensorArray(size=self.max_char_len, dtype=tf.float32, name='mean_array')
        sigma_ta = tf.TensorArray(size=self.max_char_len, dtype=tf.float32, name='sigma_array')
        word_pos = tf.convert_to_tensor(word_pos, dtype=tf.float32)
        word_locations = tf.convert_to_tensor(word_locations, dtype=tf.float32)

        # create loop_fn for raw_rnn
        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None if time = 0

            if cell_output is None:  # time = 0
                next_cell_state = cell.zero_state(self.batch_size, tf.float32)
                sample_loop_state = output_ta
                mean_loop_state = mean_ta
                sigma_loop_state = sigma_ta
                next_loop_state = (sample_loop_state, mean_loop_state, sigma_loop_state)
            # next_input = tf.zeros(shape=[self.batch_size,self.input_size],dtype=tf.float32)

            else:

                comp_tensor = tf.multiply(tf.ones([self.batch_size], dtype=tf.float32),
                                          tf.cast(time - 1, dtype=tf.float32))
                sample_mask = tf.cast(tf.equal(word_locations, comp_tensor), dtype=tf.float32)
                other_words_mask = word_pos[:, time - 1] - sample_mask
                sample_mask = tf.cast(tf.transpose(
                    tf.reshape(tf.tile(sample_mask, [self.hidden_size]), [self.hidden_size, self.batch_size]),
                    perm=[1, 0]), dtype=tf.float32)
                other_words_mask = tf.cast(tf.transpose(
                    tf.reshape(tf.tile(other_words_mask, [self.hidden_size]), [self.hidden_size, self.batch_size]),
                    perm=[1, 0]), dtype=tf.float32)

                word_slice = tf.transpose(
                    tf.reshape(tf.tile(word_pos[:, time - 1], [self.hidden_size]), [self.hidden_size, self.batch_size]),
                    perm=[1, 0])
                next_sampled_input = tf.multiply(cell_output, word_slice)

                # reparametrization
                z_concat = tf.contrib.layers.fully_connected(next_sampled_input, 2 * self.hidden_size,
                                                             activation_fn=None)
                z_mean = z_concat[:, :self.hidden_size]
                z_mean = z_mean * 10
                z_log_sigma_sq = z_concat[:, self.hidden_size:self.hidden_size * 2]
                z_log_sigma_sq = z_log_sigma_sq - 3
                eps = tf.random_normal((self.batch_size, self.hidden_size), 0, 1, dtype=tf.float32)
                z_sample = tf.add(z_mean, tf.multiply(tf.exp(z_log_sigma_sq), eps))

                z_sample = tf.multiply(z_sample, word_slice)
                z_mean = tf.multiply(z_mean, word_slice)
                z_log_sigma_sq = tf.multiply(z_log_sigma_sq, word_slice)

                next_cell_state = tf.multiply(z_sample, sample_mask) + tf.multiply(z_mean, other_words_mask)

                sample_loop_state = loop_state[0].write(time - 1, next_cell_state)
                mean_loop_state = loop_state[1].write(time - 1, z_mean)
                sigma_loop_state = loop_state[2].write(time - 1, z_log_sigma_sq)
                next_loop_state = (sample_loop_state, mean_loop_state, sigma_loop_state)

                word_slice = tf.logical_not(tf.cast(word_slice, dtype=tf.bool))
                word_slice = tf.cast(word_slice, dtype=tf.float32)
                next_cell_state = next_cell_state + tf.multiply(cell_state[0], word_slice)
                next_cell_state = tf.contrib.rnn.LSTMStateTuple(next_cell_state, cell_output)

            next_input = tf.cond(time < self.max_char_len, lambda: _inputs_ta.read(time),
                                 lambda: tf.zeros(shape=[self.batch_size, self.input_size], dtype=tf.float32))

            elements_finished = (time >= (self.max_char_len))

            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        with tf.variable_scope('encoder_rnn', reuse=reuse):
            outputs_ta, final_state_out, word_state = tf.nn.raw_rnn(cell, loop_fn)

        word_state_out = word_state[0].stack()
        mean_state_out = word_state[1].stack()
        sigma_state_out = word_state[2].stack()
        outputs_out = outputs_ta.stack()

        return word_state_out, mean_state_out, sigma_state_out


if __name__ == "__main__":

    data, eow_loc_all, _, _, _, _, _ = run_preprocess(mode="train")
    print(len(data))
    max_char_len = 371
    batch_size = 52
    input_size = 61
    hidden_size = 20
    sample_pos = 2
    num_batches = len(data) // batch_size
    arg_dict = {'max_char_len': max_char_len, 'batch_size': batch_size, 'input_size': input_size,
                'hidden_size': hidden_size}
    encoder = Encoder(**arg_dict)

    # placeholders
    inputs = tf.placeholder(tf.float32, [batch_size, max_char_len, input_size])
    word_pos = tf.placeholder(tf.float32, [batch_size, max_char_len])
    word_locations = tf.placeholder(tf.float32, [batch_size])

    word_state_out, mean_state_out, sigma_state_out = encoder.run_encoder(inputs, word_pos, word_locations, reuse=None)

    # example
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run([init_op])
        for epoch in range(1):
            epoch_error = 0

            for bt in range(1):
                x = data[bt * batch_size:(bt + 1) * batch_size]
                word_pos_batch = eow_loc_all[bt * batch_size:(bt + 1) * batch_size]
                word_loc = eow_pos(word_pos_batch, sample_pos, batch_size, max_char_len)
                word_state, mean_state, sigma_state = sess.run([word_state_out, mean_state_out, sigma_state_out],
                                                               feed_dict={inputs: x, word_pos: word_pos_batch,
                                                                          word_locations: word_loc})

                print(word_state[0, :])
                for j in range(20):
                    print(sigma_state[5, :, j])
