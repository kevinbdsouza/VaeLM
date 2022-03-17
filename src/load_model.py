import tensorflow as tf

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('/Users/kevindsouza/Documents/UBC/Research/ML/VaeLM/model/saved/vae-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()

# train operations
train_logits = graph.get_tensor_by_name("train_logits:0")
anneal_value = graph.get_tensor_by_name("anneal_value:0")
train_cost = graph.get_tensor_by_name("train_cost:0")
train_step = graph.get_tensor_by_name("train_step:0")
global_step = graph.get_tensor_by_name("global_step:0")
tf_summary_train = graph.get_tensor_by_name("tf_summary_train:0")

# train placeholders
mask_kl_pl = graph.get_tensor_by_name("kl_pl_mask:0")
onehot_words_pl = graph.get_tensor_by_name("onehot_words:0")
word_pos_pl = graph.get_tensor_by_name("word_pos:0")
perm_mat_pl = graph.get_tensor_by_name("perm_mat_pl:0")
sent_word_len_list_pl = graph.get_tensor_by_name("word_lens:0")
sent_char_len_list_pl = graph.get_tensor_by_name("sent_char_len_list:0")

# test operations
test_logits = graph.get_tensor_by_name("test_logits:0")
test_cost = graph.get_tensor_by_name("test_cost:0")
tf_summary_test = graph.get_tensor_by_name("tf_summary_test:0")

# test placeholders
onehot_words_pl_val = graph.get_tensor_by_name("onehot_words_val:0")
word_pos_pl_val = graph.get_tensor_by_name("word_pos_val:0")
perm_mat_pl_val = graph.get_tensor_by_name("perm_mat_val:0")
sent_word_len_list_pl_val = graph.get_tensor_by_name("word_lens_val:0")
sent_char_len_list_pl_val = graph.get_tensor_by_name("sent_char_len_list_val:0")

print("done")
