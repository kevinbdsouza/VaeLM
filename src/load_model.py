import tensorflow as tf


sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('/Users/kevindsouza/Documents/UBC/Research/ML/VaeLM/model/saved/vae-model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()

#train
train_logits = graph.get_tensor_by_name("train_logits:0")
anneal_value = graph.get_tensor_by_name("anneal_value:0")
train_cost = graph.get_tensor_by_name("train_cost:0")
train_step = graph.get_tensor_by_name("train_step:0")
global_step = graph.get_tensor_by_name("global_step:0")
tf_summary_train = graph.get_tensor_by_name("tf_summary_train:0")

#test
test_logits = graph.get_tensor_by_name("test_logits:0")
test_cost = graph.get_tensor_by_name("test_cost:0")
tf_summary_test = graph.get_tensor_by_name("tf_summary_test:0")


print("done")
