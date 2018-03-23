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
    return perm_mat, max_word_len,word_len_list


def permute_encoder_output(encoder_out, perm_mat, batch_size, max_word_len):
    """assumes input to function is time major, this is all in tensorflow"""
    # why do these all start at 0?
    # replace 0's possibly with len+1
    ## RELYING ON THERE BEING NOTHING AT ZEROS
    o = tf.transpose(encoder_out, perm=[1, 0, 2])
    # just to be sure
    perm_mat = tf.reshape(perm_mat, [batch_size, max_word_len])
    # permutations done instance by instance
    #elems = tf.stack([tf.unstack(o, axis=0), tf.cast(perm_mat,dtype=tf.int32)])
    #o=tf.map_fn(fn= lambda elems: tf.gather(params=elems[0], indices=tf.cast(elems[1],dtype=tf.int32), axis=0),elems=elems, dtype=(tf.float32,tf.float32) )
    o = tf.stack([tf.gather(params=i, indices=j, axis=0) for i, j in zip(tf.unstack(o, axis=0), tf.unstack(perm_mat,axis=0))])
    return o


#def train(n_epochs,**kwargs):
onehot_words,word_pos,sentence_lengths_nchars,sentence_lengths_nwords,vocabulary_size = encoder.run_preprocess()
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
sent_char_len_list_pl= tf.placeholder(name='input',dtype=tf.float32,shape=[batch_size])


word_state_out, mean_state_out, logsig_state_out = encoder_k.run_encoder(inputs=onehot_words_pl, word_pos=word_pos_pl,reuse=None)

#picking out our words
#why do these all start at 0?
# replace 0's possibly with len+1
## RELYING ON THERE BEING NOTHING AT ZEROS
word_state_out.set_shape([max_char_len,batch_size,hidden_size])
mean_state_out.set_shape([max_char_len,batch_size,hidden_size])
logsig_state_out.set_shape([max_char_len,batch_size,hidden_size])
word_state_out_p = permute_encoder_output(encoder_out=word_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
mean_state_out_p = permute_encoder_output(encoder_out=mean_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
logsig_state_out_p = permute_encoder_output(encoder_out=logsig_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
#Initialize decoder
##Note to self: need to input sentence lengths vector, also check to make sure all the placeholders flow into my class and tensorflow with ease
arg_dict = {'global_lat_dim':hidden_size,'word_lens':word_len_list_pl,'batch_size':batch_size,'max_num_words':max_word_len,'decoder_units':decoder_dim,'sentence_lens':sent_len_list_pl,'num_sentence_characters':max_char_len,'dict_length':vocabulary_size}
decoder = decoder.Decoder(**arg_dict)
out_o, global_latent_o,global_logsig_o,global_mu_o = decoder.run_decoder(units_lstm_decoder=decoder_dim,lat_words=word_state_out_p,units_dense_global=decoder_dim,sequence_length=tf.cast(sent_char_len_list_pl,dtype=tf.int32))

cost = decoder.calc_cost(global_latent_sample=global_latent_o,global_logsig=global_logsig_o,global_mu=global_mu_o,predictions=out_o,true_input=onehot_words_pl,posterior_logsig=logsig_state_out_p,posterior_mu=mean_state_out_p,post_samples=word_state_out_p)

onehot_words = np.reshape(onehot_words,newshape=[-1,batch_size,max_char_len,vocabulary_size])
word_pos = np.reshape(word_pos,newshape=[-1,batch_size,max_char_len])
sentence_lengths_nwords = np.reshape(sentence_lengths_nwords,newshape=[-1,batch_size])
sentence_lengs_nchars = np.reshape(sentence_lengs_nchars,newshape=[-1,batch_size])
######
# Train Step

# clipping gradients
######

######
#testing stuff

word_state_out, mean_state_out, logsig_state_out = encoder_k.run_encoder(inputs=onehot_words_pl, word_pos=word_pos_pl,reuse=None)
word_state_out.set_shape([max_char_len,batch_size,hidden_size])
mean_state_out.set_shape([max_char_len,batch_size,hidden_size])
logsig_state_out.set_shape([max_char_len,batch_size,hidden_size])
word_state_out_p = permute_encoder_output(encoder_out=word_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
mean_state_out_p = permute_encoder_output(encoder_out=mean_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
logsig_state_out_p = permute_encoder_output(encoder_out=logsig_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
out_o, global_latent_o,global_logsig_o,global_mu_o = decoder.run_decoder(units_lstm_decoder=decoder_dim,lat_words=word_state_out_p,units_dense_global=decoder_dim,sequence_length=tf.cast(sent_char_len_list_pl,dtype=tf.int32))
###KL annealing parameters
shift = 0
total_steps = (n_epochs/2)*np.shape(onehot_words)[0]

####
cost = decoder.calc_cost(shift=shift,total_steps=total_steps,global_step=global_step,global_latent_sample=global_latent_o,global_logsig=global_logsig_o,global_mu=global_mu_o,predictions=out_o,true_input=onehot_words_pl,posterior_logsig=logsig_state_out_p,posterior_mu=mean_state_out_p,post_samples=word_state_out_p)

######

######
#prior sampling

######
for epoch in range(n_epochs):
    inds = range(np.shape(onehot_words)[0])
    np.random.shuffle(inds)
    for count,batch in enumerate(inds):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            o=sess.run(word_state_out_p,feed_dict={onehot_words_pl:onehot_words[batch],word_pos_pl:word_pos[batch],perm_mat_pl:perm_mat[batch],word_len_list_pl:sentence_lengths_nwords[batch],sent_char_len_list_pl:sentence_lengs_nchars[batch]})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o=sess.run(out_o,feed_dict={onehot_words_pl:onehot_words[0:52],word_pos_pl:word_pos[0:52],perm_mat_pl:perm_mat[0],word_len_list_pl:sentence_lengths_nwords[0:52],sent_char_len_list_pl:sentence_lengs_nchars[0:52]})

##
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o=sess.run(word_state_out_p,feed_dict={onehot_words_pl:onehot_words[0:52],word_pos_pl:word_pos[0:52],perm_mat_pl:perm_mat[0]})


word_state_out_p = permute_encoder_output(encoder_out=input_pl, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o2=sess.run(word_state_out_p,feed_dict={perm_mat_pl:perm_mat[0], input_pl:o})