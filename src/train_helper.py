import tensorflow as tf
import numpy as np

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

    o = tf.transpose(encoder_out, perm=[1, 0, 2])
    perm_mat = tf.reshape(perm_mat, [batch_size, max_word_len])
    o = tf.stack([tf.gather(params=i, indices=j, axis=0) for i, j in zip(tf.unstack(o, axis=0), tf.unstack(perm_mat,axis=0))])
    return o

def kl_mask_prep(lat_sent_len_list,lat_sent_len_list_val,max_lat_word_len,batch_size):
    # making word masks for kl term
    kl_mask = []
    for word_len in np.reshape(lat_sent_len_list, -1):
        vec = np.zeros([max_lat_word_len], dtype=np.float32)
        vec[0:word_len] = np.ones(shape=word_len, dtype=np.float32)
        kl_mask.append(vec)
    kl_mask = np.asarray(kl_mask)
    kl_mask = np.reshape(kl_mask, newshape=[-1, batch_size, max_lat_word_len])

    kl_mask_val = []
    for word_len in np.reshape(lat_sent_len_list_val, -1):
        vec = np.zeros([max_lat_word_len], dtype=np.float32)
        vec[0:word_len] = np.ones(shape=word_len, dtype=np.float32)
        kl_mask_val.append(vec)
    kl_mask_val = np.asarray(kl_mask_val)
    kl_mask_val = np.reshape(kl_mask_val, newshape=[-1, batch_size, max_lat_word_len])

    return kl_mask,kl_mask_val