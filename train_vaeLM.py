from decoder import Decoder
import encoder
import tensorflow as tf
import numpy as np


def prep_perm_matrix(batch_size, word_pos_matrix, max_char_len):
    word_pos_matrix = np.reshape(word_pos_matrix, [-1, batch_size, max_char_len])
    full_perm_mat = []
    len_list = []
    for batch in word_pos_matrix:
        perm_mat = []
        for k, l in enumerate(batch):
            inds = np.where(l == 1)[0]
            perm_mat.append(inds)
            len_list.append(len(inds))

        full_perm_mat.append(perm_mat)

    max_word_len = np.max(len_list)

    perm_mat = []
    word_len_list=[]
    for _, batch_perm in enumerate(full_perm_mat):
        batch_perm_mat = np.zeros([batch_size, max_word_len], dtype=np.int32)

        for bnum, sentence_perm in enumerate(batch_perm):
            batch_perm_mat[bnum, 0:len(sentence_perm)] = sentence_perm
            word_len_list.append(len(sentence_perm))

        perm_mat.append(batch_perm_mat)

    perm_mat = np.asarray(perm_mat)
    return perm_mat,word_lens, max_word_len,word_len_list


def permute_encoder_output(encoder_out, perm_mat, batch_size, max_word_len):
    """assumes input to function is time major, this is all in tensorflow"""
    # why do these all start at 0?
    # replace 0's possibly with len+1
    ## RELYING ON THERE BEING NOTHING AT ZEROS
    o = tf.transpose(encoder_out, axis=[1, 0, 2])
    # just to be sure
    perm_mat = tf.reshape(perm_mat, [batch_size, max_word_len])
    # permutations done instance by instance
    o = tf.stack([tf.gather(params=i, indices=j, axis=0) for i, j in zip(tf.unstack(o, axis=0), perm_mat)])
    return o


#def train(n_epochs,**kwargs):
onehot_words,word_pos,vocabulary_size = encoder.run_preprocess()
max_char_len = 494 #kwargs['max_char_len']
batch_size = 52 #kwargs['batch_size']
input_size = vocabulary_size
hidden_size = 20 #kwargs['hidden_size']
decoder_dim = 20 #kwargs['decoder_dim']
num_batches = len(onehot_words) // batch_size

arg_dict = {'max_char_len': max_char_len, 'batch_size': batch_size, 'input_size': input_size,'hidden_size': hidden_size}
encoder_k = encoder.Encoder(**arg_dict)
#onehot_words,word_pos,vocabulary_size = encoder_k.run_preprocess()
#prepping permutation matrix for all instances seperately
perm_mat,max_word_len,sent_len_list = prep_perm_matrix(batch_size=batch_size,word_pos_matrix=word_pos,max_char_len=max_char_len)

#placeholders
word_len_list_pl  = tf.placeholder(name='word_lens',dtype=tf.int32,shape=[batch_size])
perm_mat_pl = tf.placeholder(name='perm_mat_pl',dtype=tf.int32,shape=[batch_size,max_word_len])
onehot_words_pl =tf.placeholder(name='input',dtype=tf.float32,shape=[batch_size, max_char_len, vocabulary_size])
word_pos_pl =tf.placeholder(name='input',dtype=tf.float32,shape=[batch_size, max_char_len])



word_state_out, mean_state_out, logsig_state_out = encoder_k.run_encoder(inputs=onehot_words_pl, word_pos=word_pos_pl,reuse=None)

#picking out our words
#why do these all start at 0?
# replace 0's possibly with len+1
## RELYING ON THERE BEING NOTHING AT ZEROS
word_state_out_p = permute_encoder_output(encoder_out=word_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
mean_state_out_p = permute_encoder_output(encoder_out=mean_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
logsig_state_out_p = permute_encoder_output(encoder_out=logsig_state_out , perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)

#Initialize decoder
##Note to self: need to input sentence lengths vector, also check to make sure all the placeholders flow into my class and tensorflow with ease
arg_dict = {'global_lat_dim':hidden_size,'word_lens':word_len_list_pl,'batch_size':batch_size,'max_num_words':max_word_len,'decoder_units':decoder_dim,'encodings':word_state_out_p,'sentence_lens':sent_len_list_pl,'num_sentence_characters':max_char_len,'dict_length':vocabulary_size}
decoder = decoder.Decoder(**arg_dict)


