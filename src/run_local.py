from train_vaeLM import train
from create_h5 import read

def experiment(exp_name):
    onehot_words, word_pos, sentence_lens_nchars, sentence_lens_nwords, vocabulary_size, max_char_len = read(
        file_name='/Users/kevindsouza/Documents/UBC/Research/ML/VaeLM/data/english/train.h5', train=True)
    onehot_words_val, word_pos_val, sentence_lens_nchars_val, sentence_lens_nwords_val, _, _ = read(
        file_name='/Users/kevindsouza/Documents/UBC/Research/ML/VaeLM/data/english/test.h5', train=False)

    max_char_len = 371
    batch_size = 20
    hidden_size = 256
    decoder_dim = 256
    decoder_units_p3 = 256
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

    network_dict = {'vocabulary_size': vocabulary_size, 'max_char_len': max_char_len, 'batch_size': batch_size,
                    'hidden_size': hidden_size}

    print("start")
    train(log_dir='/Users/kevindsouza/Documents/UBC/Research/ML/VaeLM/log/', n_epochs=500, network_dict=network_dict,
          index2token=index2token,mode = "half_trained", **train_dict)
    print("end")


if __name__ == '__main__':
    exp_name = 'vae_model'

    experiment(exp_name)
