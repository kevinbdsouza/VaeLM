import numpy as np
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