from cluster import WorkerPool
import h5py

def experiment(exp_name):
	from decoder import Decoder
	import encoder
	import tensorflow as tf
	import numpy as np
	from functools import reduce
	import tensorflow as tf
	import numpy as np
	from tensorflow.python.ops.rnn import _transpose_batch_time
	import collections
	import os
	import argparse
	import datetime as dt
	from collections import Counter
	from random import random
	from nltk import word_tokenize
	import train_vaeLM as *

	max_char_len = 494
	batch_size = 52
	hidden_size = 1024
	decoder_dim = 1024

	train_dict={'max_char_len':494,'batch_size':52,'hidden_size':hidden_size,'decoder_dim':decoder_dim}
	network_dict = {'max_char_len': max_char_len, 'batch_size': batch_size,'hidden_size': hidden_size}


	log("start")
	train(n_epochs=1,network_dict=network_dict,**train_dict)
	log("end")


if __name__ == '__main__':

    exp_name = 'vae_model'



    pool = WorkerPool(1, worker_types='gpu_gtx_1080_ti')

    pool.attach_files(['train_vaeLM.py','decoder.py','encoder.py','cluster_print.py',h5py])

    pool(experiment, [exp_name])

