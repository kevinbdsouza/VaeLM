import encoder
import h5py
import numpy as np
from sys import argv


def write(file_name,list_inp,train):
    onehot_words, word_pos, sentence_lens_nchars, sentence_lens_nwords, vocabulary_size, index2token,max_char_len,word_loc = list_inp

    with h5py.File(file_name, "w") as f:
    	if train:
            	g_train = f.create_group('train_group')
            	g_train['onehot_words'] = onehot_words
            	g_train['word_pos'] = word_pos
            	g_train['sentence_lens_nchars'] = sentence_lens_nchars
            	g_train['sentence_lens_nwords'] = sentence_lens_nwords
            	g_train['vocabulary_size'] = vocabulary_size
            	#g_train['index2token'] = index2token
            	g_train['max_char_len'] = max_char_len
                g_train['word_loc'] = word_loc

    	else:
            	g_val = f.create_group('val_group')
            	g_val['onehot_words'] = onehot_words
            	g_val['word_pos'] = word_pos
            	g_val['sentence_lens_nchars'] = sentence_lens_nchars
            	g_val['sentence_lens_nwords'] = sentence_lens_nwords
            	g_val['vocabulary_size'] = vocabulary_size
            	#g_val['index2token'] = index2token_val
            	g_val['max_char_len'] = max_char_len
                g_val['word_loc'] = word_loc


def read(file_name,train=True):

    list_inp=[]
    with h5py.File(file_name, "r") as f:
        if train:
            g_train = f.get('train_group')

            list_inp.append(g_train['onehot_words'].value)
            list_inp.append(g_train['word_pos'].value)
            list_inp.append(g_train['sentence_lens_nchars'].value)
            list_inp.append(g_train['sentence_lens_nwords'].value)
            list_inp.append(g_train['vocabulary_size'].value)
            #list_inp.append(g_train['index2token'].value)
            list_inp.append(g_train['max_char_len'].value)
            list_inp.append(g_train['word_loc'].value)

        else:
            g_val = f.get('val_group')

            list_inp.append(g_val['onehot_words'].value)
            list_inp.append(g_val['word_pos'].value)
            list_inp.append(g_val['sentence_lens_nchars'].value)
            list_inp.append(g_val['sentence_lens_nwords'].value)
            list_inp.append(g_val['vocabulary_size'].value)
            #list_inp.append(g_val['index2token'].value)
            list_inp.append(g_val['max_char_len'].value)
            list_inp.append(g_val['word_loc'].value)

    #onehot_words,word_pos,sentence_lens_nchars,sentence_lens_nwords,vocabulary_size,index2token,max_char_len = list_inp

    return list_inp
if __name__=='__main__':

    #python arguments expected from argv, i.e. just the name of the file and directory train_file.h5, valid_file.h5
    sample_pos = 2
    onehot_words, word_pos, sentence_lens_nchars, sentence_lens_nwords, vocabulary_size, index2token, max_char_len = encoder.run_preprocess(mode='train')
    onehot_words_val, word_pos_val, sentence_lens_nchars_val, sentence_lens_nwords_val, vocabulary_size_val, index2token_val, max_char_len = encoder.run_preprocess(mode='val')

    word_loc = encoder.eow_pos(word_pos,sample_pos,len(onehot_words),max_char_len)
    word_loc_val = encoder.eow_pos(word_pos,sample_pos,len(onehot_words_val),max_char_len)

    l_train = [onehot_words, word_pos, sentence_lens_nchars, sentence_lens_nwords, vocabulary_size, index2token,
               max_char_len,word_loc]
    l_val = [onehot_words_val, word_pos_val, sentence_lens_nchars_val, sentence_lens_nwords_val, vocabulary_size_val,
             index2token_val, max_char_len,word_loc_val]

    write(file_name=argv[-2],list_inp=l_train,train=True)
    write(file_name=argv[-1], list_inp=l_val,train=False)
