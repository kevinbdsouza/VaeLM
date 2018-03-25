from decoder import Decoder
import encoder
import tensorflow as tf
import numpy as np
from functools import reduce

def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))

def prep_perm_matrix(batch_size, word_pos_matrix, max_char_len,max_word_len=None):
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

    if max_word_len is None:
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

def train(n_epochs,network_dict,**kwargs):
    onehot_words,word_pos,sentence_lens_nchars,sentence_lens_nwords,vocabulary_size,index2token,max_char_len = encoder.run_preprocess(mode='train')
    onehot_words_val,word_pos_val,sentence_lens_nchars_val,sentence_lens_nwords_val,vocabulary_size_val,index2token_val,max_char_len = encoder.run_preprocess(mode='val')

    #max_char_len = kwargs['max_char_len']
    batch_size = kwargs['batch_size']
    input_size = vocabulary_size
    hidden_size = kwargs['hidden_size']
    decoder_dim = kwargs['decoder_dim']
    num_batches = len(onehot_words) // batch_size
    network_dict['input_size'] = input_size


    encoder_k = encoder.Encoder(**network_dict)


    #onehot_words,word_pos,vocabulary_size = encoder_k.run_preprocess()
    #prepping permutation matrix for all instances seperately
    perm_mat,max_word_len,sent_len_list = prep_perm_matrix(batch_size=batch_size,word_pos_matrix=word_pos,max_char_len=max_char_len)

    #placeholders
    sent_word_len_list_pl  = tf.placeholder(name='word_lens',dtype=tf.int32,shape=[batch_size])
    perm_mat_pl = tf.placeholder(name='perm_mat_pl',dtype=tf.int32,shape=[batch_size,max_word_len])
    onehot_words_pl =tf.placeholder(name='onehot_words',dtype=tf.float32,shape=[batch_size, max_char_len, vocabulary_size])
    word_pos_pl =tf.placeholder(name='word_pos',dtype=tf.float32,shape=[batch_size, max_char_len])
    sent_char_len_list_pl= tf.placeholder(name='sent_char_len_list',dtype=tf.float32,shape=[batch_size])
    #decoder
    arg_dict = {'encoder_dim':hidden_size,'lat_word_dim':hidden_size,'sentence_lens':None,'global_lat_dim':hidden_size,'batch_size':batch_size,'max_num_words':max_word_len,'decoder_units':decoder_dim,'num_sentence_characters':max_char_len,'dict_length':vocabulary_size}
    decoder = Decoder(**arg_dict)


    #step counter
    global_step = tf.Variable(0, name='global_step', trainable=False)


    word_state_out, mean_state_out, logsig_state_out = encoder_k.run_encoder(inputs=onehot_words_pl, word_pos=word_pos_pl,reuse=None)

    #picking out our words
    #why do these all start at 0?
    # replace 0's possibly with len+1
    ## RELYING ON THERE BEING NOTHING AT ZEROS
    #indice 0 problem?
    word_state_out.set_shape([max_char_len,batch_size,hidden_size])
    mean_state_out.set_shape([max_char_len,batch_size,hidden_size])
    logsig_state_out.set_shape([max_char_len,batch_size,hidden_size])
    word_state_out_p = permute_encoder_output(encoder_out=word_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
    mean_state_out_p = permute_encoder_output(encoder_out=mean_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
    logsig_state_out_p = permute_encoder_output(encoder_out=logsig_state_out, perm_mat=perm_mat_pl, batch_size=batch_size, max_word_len=max_word_len)
    #Initialize decoder
    ##Note to self: need to input sentence lengths vector, also check to make sure all the placeholders flow into my class and tensorflow with ease

    out_o, global_latent_o,global_logsig_o,global_mu_o = decoder.run_decoder(reuse=None,units_lstm_decoder=decoder_dim,lat_words=word_state_out_p,units_dense_global=decoder_dim,sequence_length=tf.cast(sent_char_len_list_pl,dtype=tf.int32))

    # shaping for batching
    #reshape problem
    onehot_words = np.reshape(onehot_words,newshape=[-1,batch_size,max_char_len,vocabulary_size])
    word_pos = np.reshape(word_pos,newshape=[-1,batch_size,max_char_len])
    sentence_lens_nwords = np.reshape(sentence_lens_nwords,newshape=[-1,batch_size])
    sentence_lens_nchars = np.reshape(sentence_lens_nchars,newshape=[-1,batch_size])

    #shaping for validation set
    batch_size_val = batch_size
    n_valid = np.shape(onehot_words_val)[0]

    r = n_valid%batch_size_val
    n_valid_use= n_valid-r
    #might have to fix this before reporting results
    onehot_words_val = np.reshape(onehot_words_val[0:n_valid_use,...],newshape=[-1,batch_size_val,max_char_len,vocabulary_size])
    word_pos_val = np.reshape(word_pos_val[0:n_valid_use,...],newshape=[-1,batch_size_val,max_char_len])
    sentence_lens_nwords_val = np.reshape(sentence_lens_nwords_val[0:n_valid_use],newshape=[-1,batch_size_val])
    sentence_lens_nchars_val = np.reshape(sentence_lens_nchars_val[0:n_valid_use],newshape=[-1,batch_size_val])

    ###KL annealing parameters
    shift = 0
    total_steps = np.round(np.true_divide(n_epochs,2)*np.shape(onehot_words)[0],decimals=0)

    ####
    cost,reconstruction,kl_p3,kl_p1,kl_global,kl_p2,anneal = decoder.calc_cost(sentence_word_lens=sent_word_len_list_pl,shift=shift,total_steps=total_steps,global_step=global_step,global_latent_sample=global_latent_o,global_logsig=global_logsig_o,global_mu=global_mu_o,predictions=out_o,true_input=onehot_words_pl,posterior_logsig=logsig_state_out_p,posterior_mu=mean_state_out_p,post_samples=word_state_out_p,reuse=None)

    ######
    # Train Step

    # clipping gradients
    ######
    lr = 1e-4
    opt = tf.train.AdamOptimizer(lr)
    grads_t, vars_t = zip(*opt.compute_gradients(cost))
    clipped_grads_t, grad_norm_t = tf.clip_by_global_norm(grads_t, clip_norm=10.0)
    train_step = opt.apply_gradients(zip(clipped_grads_t, vars_t), global_step=global_step)
    ######
    #testing stuff
    #testing pls
    sent_word_len_list_pl_val  = tf.placeholder(name='word_lens_val',dtype=tf.int32,shape=[batch_size])
    perm_mat_pl_val = tf.placeholder(name='perm_mat_val',dtype=tf.int32,shape=[batch_size,max_word_len])
    onehot_words_pl_val =tf.placeholder(name='onehot_words_val',dtype=tf.float32,shape=[batch_size, max_char_len, vocabulary_size])
    word_pos_pl_val =tf.placeholder(name='word_pos_val',dtype=tf.float32,shape=[batch_size, max_char_len])
    sent_char_len_list_pl_val= tf.placeholder(name='sent_char_len_list_val',dtype=tf.float32,shape=[batch_size])

    #testing graph
    word_state_out_val, mean_state_out_val, logsig_state_out_val = encoder_k.run_encoder(inputs=onehot_words_pl_val, word_pos=word_pos_pl_val,reuse=True)
    perm_mat_val,_,sent_len_list_val = prep_perm_matrix(batch_size=batch_size_val,word_pos_matrix=word_pos_val,max_char_len=max_char_len,max_word_len=max_word_len)

    word_state_out_val, mean_state_out_val, logsig_state_out_val = encoder_k.run_encoder(inputs=onehot_words_pl_val, word_pos=word_pos_pl_val,reuse=True)
    word_state_out_val.set_shape([max_char_len,batch_size_val,hidden_size])
    mean_state_out_val.set_shape([max_char_len,batch_size_val,hidden_size])
    logsig_state_out.set_shape([max_char_len,batch_size_val,hidden_size])
    word_state_out_p_val = permute_encoder_output(encoder_out=word_state_out_val, perm_mat=perm_mat_pl_val, batch_size=batch_size_val, max_word_len=max_word_len)
    mean_state_out_p_val = permute_encoder_output(encoder_out=mean_state_out_val, perm_mat=perm_mat_pl_val, batch_size=batch_size_val, max_word_len=max_word_len)
    logsig_state_out_p_val = permute_encoder_output(encoder_out=logsig_state_out_val, perm_mat=perm_mat_pl_val, batch_size=batch_size_val, max_word_len=max_word_len)
    out_o_val, global_latent_o_val,global_logsig_o_val,global_mu_o_val = decoder.run_decoder(reuse=True,units_lstm_decoder=decoder_dim,lat_words=word_state_out_p_val,units_dense_global=decoder.global_lat_dim,sequence_length=tf.cast(sent_char_len_list_pl_val,dtype=tf.int32))
    #test cost
    test_cost = decoder.test_calc_cost(sentence_word_lens=sent_word_len_list_pl_val,posterior_logsig=logsig_state_out_p_val,post_samples=word_state_out_p_val,global_mu=global_mu_o_val,global_logsig=global_logsig_o_val,global_latent_sample=global_latent_o_val,posterior_mu=mean_state_out_p_val,true_input=onehot_words_pl_val,predictions=out_o_val)

    ######

    ######
    #prior sampling
    samples = np.random.normal(size = [batch_size,decoder.global_lat_dim])
    gen_samples = decoder.generation(samples=samples)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    ######
    for epoch in range(n_epochs):
        inds = range(np.shape(onehot_words)[0])
        np.random.shuffle(inds)
        for count,batch in enumerate(inds):
            train_predictions_o_np, train_cost_o_np, _, global_step_o_np,train_rec_cost_o_np,_,_,_,_,anneal_constant=sess.run([out_o,cost,train_step,global_step,reconstruction,kl_p3,kl_p1,kl_global,kl_p2,anneal],feed_dict={onehot_words_pl:onehot_words[batch],word_pos_pl:word_pos[batch],perm_mat_pl:perm_mat[batch],sent_word_len_list_pl:sentence_lens_nwords[batch],sent_char_len_list_pl:sentence_lens_nchars[batch]})
            print('train cost: {}'.format(train_cost_o_np))
            if count % 1000:
                # testing on the validation set
                val_predictions_o_np, val_cost_o_np = sess.run(
                    [out_o_val, test_cost], feed_dict={onehot_words_pl_val: onehot_words_val[0], word_pos_pl_val: word_pos_val[0],
                                         perm_mat_pl_val: perm_mat_val[0], sent_word_len_list_pl_val: sentence_lens_nwords_val[0],
                                         sent_char_len_list_pl_val: sentence_lens_nchars_val[0]})
                print('validation cost {}'.format(out_o_val))
            if count % 10000:
                # testing on the generative model
                gen_o_np = sess.run([gen_samples])

    sess.close()



max_char_len = 494
batch_size = 52
hidden_size = 20
decoder_dim = 20

train_dict={'max_char_len':494,'batch_size':52,'hidden_size':20,'decoder_dim':20}
network_dict = {'max_char_len': max_char_len, 'batch_size': batch_size,'hidden_size': hidden_size}

train(n_epochs=1,network_dict=network_dict,**train_dict)