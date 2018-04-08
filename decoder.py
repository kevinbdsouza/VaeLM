import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import _transpose_batch_time

class Decoder:
    def __init__(self,**kwargs):
        self.encodings =None
        self.num_sentence_characters = kwargs['num_sentence_characters']
        self.dict_length = kwargs['dict_length']
        self.max_num_lat_words= kwargs['max_num_lat_words']
        self.batch_size=kwargs['batch_size']
        self.simple_decoder = True
        self.global_lat_decoder=False
        self.decoder_units=kwargs['decoder_units']
        self.units_encoder_lstm = kwargs['encoder_dim']
        self.lat_word_dim= kwargs['lat_word_dim']
        self.global_lat_dim =kwargs['global_lat_dim']
	self.decoder_p3_units= kwargs['decoder_p3_units']
    def make_global_latent(self,values,units_dense):
        mean_pool= tf.reduce_mean(values,axis=-1)
        pre_dist1 = tf.layers.dense(inputs=mean_pool,activation=tf.nn.relu, units=units_dense)
        pre_dist2 = tf.layers.dense(inputs=pre_dist1,activation=None,units=units_dense*2)
        mu, log_sig = tf.split(tf.cast(pre_dist2,dtype=tf.float32),axis=-1,num_or_size_splits=2)
        return mu, log_sig

    def decoder1_p1(self,reuse,units_bilstm,encodings=None):
        if encodings is None:
            encodings= self.encodings

        with tf.variable_scope('decoder_p1',reuse=reuse):
            cell1 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            cell2 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            values, states = tf.nn.bidirectional_dynamic_rnn(inputs=encodings, dtype=tf.float32, cell_bw=cell1,cell_fw=cell2, sequence_length=self.sentence_lens)
        values = tf.concat(values, 2)
        return values

    def decoder2_p1(self, reuse, units_bilstm, global_latent):
        #needs some work
      #  input = [global_latent for i in range(self.num_sentence_characters)]
        with tf.variable_scope('decoder_p1',reuse=reuse):
            cell1 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            cell2 = tf.contrib.rnn.LSTMCell(num_units=units_bilstm)
            values, states = tf.nn.bidirectional_dynamic_rnn(inputs=input, dtype=tf.float32, cell_bw=cell1,cell_fw=cell2, sequence_length=tf.cast(hap_lens, tf.int32))
        values = tf.concat(values, 2)
        return values

    def bahd_attention(self,queries,values,reuse):
        with tf.variable_scope('attention_layer',reuse=reuse):
            w1 = tf.get_variable(name='query_w',shape = [self.decoder_units,self.lat_word_dim])
            w2 = tf.get_variable(name='value_w',shape = [self.lat_word_dim,self.lat_word_dim])
            v = tf.get_variable(name='v',shape=[self.lat_word_dim])
            print('here')
            conv_q = tf.reshape(tf.einsum('ij,jk->ik',queries, w1),[-1,1,self.lat_word_dim])
            print('here1')
            a_p1= tf.reshape(tf.tile(conv_q,[1,1,self.max_num_lat_words]),[self.batch_size,self.max_num_lat_words,self.lat_word_dim])
            print('here2')
            print(w2)
            print('a p1 {}'.format(a_p1))

            a_p2 = tf.einsum('ijk,kl->ijl',values,w2)
            print('a p2 {}'.format(a_p2))
            print('here3')
            out = tf.einsum('k,ijk->ij',v,tf.nn.tanh(name='combine',x=a_p1+a_p2))
            print('MAT for softmax {}'.format(out))
            out_norm = tf.nn.softmax(out,dim=-1)
            context = tf.reduce_sum(values*tf.reshape(tf.stack([out_norm for _ in range(self.lat_word_dim)],-1),[self.batch_size,self.max_num_lat_words,self.lat_word_dim]),axis=-2)
            #context2 = tf.matmul(tf.reshape(tf.diag(out_norm),[-1,self.max_num_words]),tf.transpose(values,[-1,self.max_num_words]))
            #is this the same
            #print('ALT CONTEXT {}'.format(context2))
            print('CONTEX SHAPE {}'.format(context))
            #l1 = tf.concat([context,queries],axis=-1)
	    l1 = tf.reshape(context,[self.batch_size,self.lat_word_dim])
            #l1 = tf.reshape(l1,[self.batch_size,self.lat_word_dim+self.decoder_units])
        return l1

    def decoder_p2(self,num_hidden_word_units,inputs,sequence_length,global_latent,reuse,context_dim,max_time):
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)

        cell = tf.contrib.rnn.LSTMCell(self.decoder_units)

        def loop_fn(time, cell_output, cell_state,loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(self.batch_size, tf.float32)
                next_loop_state = outputs_ta
                context = self.bahd_attention(queries=tf.zeros(shape=[self.batch_size,num_hidden_word_units],dtype=tf.float32), values=inputs, reuse=None)
               # next_input = tf.concat([tf.zeros(shape=[self.batch_size,self.lat_word_dim],dtype=tf.float32),tf.zeros(shape=[self.batch_size,self.global_lat_dim],dtype=tf.float32)],axis=-1)
		next_input =tf.zeros(shape=[self.batch_size,self.lat_word_dim+self.global_lat_dim],dtype=tf.float32)
            else:
                next_cell_state = cell_state
                context = self.bahd_attention(queries=cell_output,values=inputs,reuse=True)
                # should try passing in logits
                # should also try doing the final decoding in a seperate RNN
                # should try using a global latent vector here asap
                #prediction = tf.layers.dense(inputs=context,activation=None,units=self.dict_length)
                #took context out of decoder loop because softmax may be saturating
                next_input = tf.concat([context,global_latent],axis=-1)
                next_loop_state = loop_state.write(time-1,context)
            elements_finished = (time >= sequence_length)

            return (elements_finished, next_input, next_cell_state,emit_output, next_loop_state)

        with tf.variable_scope('decoder_p2',reuse=reuse):
            _, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
#            loop_state_out = _transpose_batch_time(loop_state_ta.stack())

        return loop_state_ta
    def decoder_p3(self,inputs,reuse,max_time,sequence_length):
#        _inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time,name='context_array')
#	_inputs_ta = _inputs_ta.unstack(tf.transpose(inputs,[1,0,2]))
	_inputs_ta = inputs
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time,name='pred_char_array')

        cell = tf.contrib.rnn.LSTMCell(self.decoder_p3_units)

        def loop_fn(time, cell_output, cell_state,loop_state):
	    next_loop_state = loop_state 
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(self.batch_size, tf.float32)
	    	next_input = tf.concat([tf.zeros(shape=[self.batch_size,self.dict_length],dtype=tf.float32),_inputs_ta.read(time)],axis=-1)
                next_loop_state = outputs_ta
            else:
		
                next_cell_state = cell_state
                prediction = tf.layers.dense(inputs=cell_output,activation=None,units=self.dict_length)
                next_loop_state = loop_state.write(time-1,prediction)
		next_input = tf.concat([prediction,_inputs_ta.read(time)],axis=-1)
        #argmax seems to be working a bit better, funny as it's not differentiable
                #next_input = tf.concat([tf.one_hot(tf.argmax(prediction, -1), depth=self.dict_length, axis=-1), _inputs_ta.read(time)],axis=-1)
            elements_finished = (time >= sequence_length-1)

            return (elements_finished, next_input, next_cell_state,emit_output, next_loop_state)

        with tf.variable_scope('decoder_p3',reuse=reuse):
            _, _, loop_ta = tf.nn.raw_rnn(cell, loop_fn)
            output = _transpose_batch_time(loop_ta.stack())
	return output
    def run_decoder(self,units_lstm_decoder,sequence_length,units_dense_global,lat_words,reuse):
        if self.simple_decoder:
            global_mu, global_logsig = self.make_global_latent(values=lat_words, units_dense=units_dense_global)
            eps = tf.random_normal(shape=[self.batch_size,units_dense_global],dtype=tf.float32)
            global_latent = eps*tf.exp(global_logsig)+global_mu
            out_2 = self.decoder_p2(sequence_length=sequence_length,num_hidden_word_units=units_lstm_decoder, inputs=lat_words, reuse=reuse,global_latent=global_latent, context_dim=units_lstm_decoder, max_time=self.num_sentence_characters)
	    out = self.decoder_p3(inputs=out_2,reuse=reuse,max_time=self.num_sentence_characters,sequence_length=sequence_length)
        return out, global_latent,global_logsig,global_mu

    def prior(self, values,num_units,global_latent,word_lens,reuse):
        global_latent= tf.transpose(tf.stack([global_latent for _ in range(self.max_num_lat_words)]),[1,0,2])
        print(' PRIOR input dim from post {}'.format(values))
        values = tf.concat([tf.zeros(shape=[self.batch_size,1, self.lat_word_dim], dtype=tf.float32),values],axis=1)
        values = values[:,0:-1,:]
        values = tf.concat([tf.cast(values,dtype=tf.float32),global_latent],axis=-1)
        print('PRIOR input dim to prior {}'.format(values))
        with tf.variable_scope('prior',reuse=reuse):
            cell = tf.contrib.rnn.LSTMCell(num_units)
            values, _ = tf.nn.dynamic_rnn(cell=cell,inputs=values,sequence_length=word_lens,dtype=tf.float32)
        with tf.variable_scope('prior/rnn', reuse=reuse):
            w = tf.get_variable(name='prior_dense_w',shape=[self.lat_word_dim,self.lat_word_dim*2],dtype=tf.float32)
            b = tf.get_variable(name='prior_dense_b',shape=self.lat_word_dim*2,dtype=tf.float32)
            out = tf.reshape(tf.matmul(tf.reshape(values,[-1,self.lat_word_dim]),w)+b,[self.batch_size,self.max_num_lat_words,self.lat_word_dim*2])

            mu,log_sig = tf.split(out,axis=-1,num_or_size_splits=2,name='prior_dense')
            print('MU{}'.format(mu))
        return [mu,log_sig]

    def cost_function(self,mask_kl,predictions,true_input,global_mu,global_logsig,prior_mu,prior_logsig,posterior_mu,posterior_logsig,shift,total_steps,global_step,kl=True):
        mask = tf.reduce_sum(true_input, -1)
        #reconstruction = tf.reduce_sum(tf.reduce_sum(-true_input*tf.log(predictions+1e-9),axis=-1),-1)
        reconstruction = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(true_input,-1),logits=predictions)*mask,-1)
        #have to be very careful of order of the mean/stddev parmeters
        #outer reduce sum for each KL term
        '''
        kl_p1 = 0.5 * (tf.reduce_sum(tf.exp(posterior_logsig - prior_logsig), axis=-1) + tf.reduce_sum(
            (posterior_mu - prior_mu) * tf.divide(1, tf.exp(prior_logsig)) * (posterior_mu - prior_mu),
            axis=-1) - tf.cast(tf.shape(posterior_mu)[-1], dtype=tf.float32) + tf.reduce_sum(
            (prior_logsig - posterior_logsig), axis=-1))
        '''
        '''
        kl_p1 = 0.5 * (tf.reduce_sum(tf.reduce_sum(tf.log(tf.square(tf.exp(prior_logsig))),axis=1) - tf.reduce_sum(tf.log(tf.square(tf.exp(posterior_logsig))),axis=1),axis=-1) - 
            tf.cast(tf.shape(posterior_mu)[-1], dtype=tf.float32) * tf.cast(tf.shape(prior_mu)[1],dtype=tf.float32) + 
            tf.reduce_sum(tf.reduce_sum(tf.divide(1,tf.square(tf.exp(prior_logsig))) * tf.square(tf.exp(posterior_logsig)),axis=-1) +  tf.reduce_sum(
            (posterior_mu - prior_mu) * tf.divide(1, tf.square(tf.exp(prior_logsig))) * (posterior_mu - prior_mu),
            axis=-1),axis=-1))'''
        kl_p1 = 0.5 * (tf.reduce_sum((tf.log(tf.square(tf.exp(prior_logsig))) -
            tf.log(tf.square(tf.exp(posterior_logsig)))), axis=-1) -
                       tf.cast(tf.shape(posterior_mu)[-1], dtype=tf.float32)+tf.reduce_sum(tf.divide(1, tf.square(tf.exp(prior_logsig))) * tf.square(tf.exp(posterior_logsig)),
                           axis=-1) + tf.reduce_sum(((posterior_mu - prior_mu) * tf.divide(1, tf.square(tf.exp(prior_logsig))) * (posterior_mu - prior_mu)),axis=-1))
        #have to mask out for padding (there will be numbers there due to the dense after the rnn)
        kl_p1=tf.reduce_sum((kl_p1*mask_kl),-1)
        '''
        kl_global_lat = 0.5 * (
        tf.reduce_sum(tf.exp(global_logsig), axis=-1) + tf.reduce_sum((global_mu * global_mu), axis=-1) - tf.cast(
            tf.shape(global_mu)[-1], dtype=tf.float32) - tf.reduce_sum(global_logsig))
        '''

        kl_global_lat = 0.5 * (-tf.reduce_sum(tf.log(tf.square(tf.exp(global_logsig))),axis=-1) - tf.cast(tf.shape(global_mu)[-1],dtype=tf.float32) + tf.reduce_sum(
            tf.square(tf.exp(global_logsig)),axis=-1) + tf.reduce_sum((global_mu * global_mu), axis=-1))

        kl_p2 = kl_p1
        #kl_p2 = tf.reduce_sum(kl_p1, -1)
        if kl:
            kl_p3 = kl_p2 + kl_global_lat
            anneal_c = tf.cast(tf.minimum(tf.maximum(tf.divide((global_step-shift),total_steps),0),1),dtype=tf.float32)
            kl_p3 = kl_p3*anneal_c
        else:
            anneal_c=0
            kl_p3 = tf.constant(0,dtype=tf.float32)
        #sum over all seperate KLs for each lat var

        cost = tf.reduce_mean(kl_p3+reconstruction)
        return cost,reconstruction,kl_p3,kl_p1,kl_global_lat,kl_p2, anneal_c

    def test_cost_function(self,predictions,mask_kl,true_input,global_mu,global_logsig,prior_mu,prior_logsig,posterior_mu,posterior_logsig):
        mask = tf.reduce_sum(true_input,-1)
        reconstruction = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(true_input, -1), logits=predictions)*mask, -1)
        #reconstruction = tf.reduce_sum(-true_input*tf.log(predictions+1e-9),axis=-1)
        #have to be very careful of order of the mean/stddev parmeters
        #outer reduce sum for each KL term
#        kl_p1 = 0.5*(tf.reduce_sum(tf.exp(posterior_logsig-prior_logsig),axis=-1)+tf.reduce_sum((posterior_mu-prior_mu)*tf.divide(1,tf.exp(prior_logsig))*(posterior_mu-prior_mu),axis=-1)-tf.cast(tf.shape(posterior_mu)[-1],dtype=tf.float32)+tf.reduce_sum((prior_logsig-posterior_logsig),axis=-1))
#        kl_global_lat = 0.5*(tf.reduce_sum(tf.exp(global_logsig),axis=-1)+ tf.reduce_sum((global_mu*global_mu),axis=-1)-tf.cast(tf.shape(global_mu)[-1],dtype=tf.float32)-tf.reduce_sum(global_logsig))
        #sum over all seperate KLs for each lat var
        '''kl_p1 = 0.5 * (tf.reduce_sum(tf.reduce_sum(tf.log(tf.square(tf.exp(prior_logsig))),axis=1) - tf.reduce_sum(tf.log(tf.square(tf.exp(posterior_logsig))),axis=1),axis=-1) -
            # the constant term is not correct as each sentence will have a different number of latent word variables,but is just a constant so shouldn't change the optimization
            tf.cast(tf.shape(posterior_mu)[-1], dtype=tf.float32) * tf.cast(tf.shape(prior_mu)[1],dtype=tf.float32) +
            tf.reduce_sum(tf.reduce_sum(tf.divide(1,tf.square(tf.exp(prior_logsig))) * tf.square(tf.exp(posterior_logsig)),axis=-1) +  tf.reduce_sum(
            (posterior_mu - prior_mu) * tf.divide(1, tf.square(tf.exp(prior_logsig))) * (posterior_mu - prior_mu),
            axis=-1) ,axis=-1))'''
        kl_p1 = 0.5 * (tf.reduce_sum((tf.log(tf.square(tf.exp(prior_logsig))) -
            tf.log(tf.square(tf.exp(posterior_logsig)))), axis=-1) -
                       tf.cast(tf.shape(posterior_mu)[-1], dtype=tf.float32)+tf.reduce_sum(tf.divide(1, tf.square(tf.exp(prior_logsig))) * tf.square(tf.exp(posterior_logsig)),
                           axis=-1) + tf.reduce_sum(((posterior_mu - prior_mu) * tf.divide(1, tf.square(tf.exp(prior_logsig))) * (posterior_mu - prior_mu)),axis=-1))
        #have to mask out for padding (there will be numbers there due to the dense after the rnn)
        kl_p1=tf.reduce_sum((kl_p1*mask_kl),-1)

        kl_global_lat = 0.5 * (-tf.reduce_sum(tf.log(tf.square(tf.exp(global_logsig))),axis=-1) - tf.cast(tf.shape(global_mu)[-1],dtype=tf.float32) + tf.reduce_sum(
            tf.square(tf.exp(global_logsig)),axis=-1) + tf.reduce_sum((global_mu * global_mu), axis=-1))

        kl_p3 = kl_p1+kl_global_lat
        cost = tf.reduce_mean(kl_p3+reconstruction)
        return cost,reconstruction,kl_p3,kl_p1

    def calc_cost(self,mask_kl,kl,posterior_logsig,post_samples,global_mu,global_logsig,global_latent_sample,posterior_mu,true_input,sentence_word_lens,predictions,shift, total_steps, global_step,reuse):
        prior_mu,prior_logsig = self.prior(values=post_samples,num_units=self.units_encoder_lstm,global_latent=global_latent_sample,word_lens=sentence_word_lens,reuse=reuse)
        cost, reconstruction, kl_p3, kl_p1,kl_global,kl_p2,anneal_c= self.cost_function(mask_kl=mask_kl,kl=kl,predictions=predictions,true_input=true_input,global_mu=global_mu,global_logsig=global_logsig,prior_mu=prior_mu,prior_logsig=prior_logsig,posterior_mu=posterior_mu,posterior_logsig=posterior_logsig,shift=shift, total_steps=total_steps, global_step=global_step)
    	self.kls_hist = tf.summary.histogram('kls',tf.reduce_mean(kl_p1,0))
    	self.global_kl_scalar = tf.summary.scalar('kls_global',tf.reduce_mean(kl_global))
    	self.rec_scalar = tf.summary.scalar('rec',tf.reduce_mean(reconstruction))
    	self.cost_scalar = tf.summary.scalar('full_cost',cost)
    	var_all = tf.nn.moments(x=posterior_mu,axes=0)
    	var_all=var_all[-1]
    	kl = tf.reduce_mean(kl_p3)

    	self.full_kl_scalar = tf.summary.scalar('full_kl',kl)
    	self.sum_all_activ_hist = tf.summary.histogram('active_lats_all', var_all)
        var_g = tf.nn.moments(x=global_mu, axes=0)
        var_g = var_g[-1]
        self.sum_global_activ_hist = tf.summary.histogram('active_lats_global', var_g)

        return cost,reconstruction,kl_p3,kl_p1,kl_global,kl_p2,anneal_c,prior_mu

    def test_calc_cost(self,mask_kl,posterior_logsig,post_samples,global_mu,global_logsig,global_latent_sample,posterior_mu,true_input,predictions,sentence_word_lens):
        prior_mu,prior_logsig = self.prior(values=post_samples,num_units=self.units_encoder_lstm,global_latent=global_latent_sample,word_lens=sentence_word_lens,reuse=True)
        cost,reconstruction,kl_p3,kl_p1 = self.test_cost_function(mask_kl=mask_kl,predictions=predictions,true_input=true_input,global_mu=global_mu,global_logsig=global_logsig,prior_mu=prior_mu,prior_logsig=prior_logsig,posterior_mu=posterior_mu,posterior_logsig=posterior_logsig)
        self.sum_rec_val = tf.summary.scalar('rec_test',tf.reduce_mean(reconstruction))
        self.sum_kl_val = tf.summary.scalar('kl_test',tf.reduce_mean(kl_p3))
        return cost

    def generation(self,samples):
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=self.max_num_lat_words)
            cell = tf.contrib.rnn.LSTMCell(self.decoder_units)
            print('GENER samples {}'.format(np.shape(samples)))



            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output  # == None for time == 0
                if cell_output is None:  # time == 0
                    next_cell_state = cell.zero_state(self.batch_size, tf.float32)
                    next_loop_state = outputs_ta

                    #self.lat_word_dim is very important, need from kevin
                    next_input = tf.concat([tf.zeros(shape=[self.batch_size,self.lat_word_dim],dtype=tf.float32),samples],axis=-1)

                else:
                    next_cell_state = cell_state
                    w = tf.get_variable(name='prior_dense_w')
                    b = tf.get_variable(name='prior_dense_b')
                    print(cell_output)
                    cell_output = tf.reshape(tf.matmul(cell_output, w) + b,[self.batch_size,self.lat_word_dim * 2])


                    mu, logsig = tf.split(cell_output,axis=-1,num_or_size_splits=2)
                    eps = tf.random_normal(shape=[self.batch_size,self.lat_word_dim],dtype=tf.float32)
                    samples_word = eps*tf.exp(logsig)+mu

                    next_input = tf.concat([samples_word,samples],axis=-1)

                    next_loop_state = loop_state.write(time - 1, samples_word)

                elements_finished = (time >= self.max_num_lat_words)

                return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

            with tf.variable_scope('prior', reuse=True):
                _, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
                loop_state_out = _transpose_batch_time(loop_state_ta.stack())
            context = self.decoder_p2(num_hidden_word_units=self.lat_word_dim, inputs=loop_state_out, sequence_length=np.repeat(self.num_sentence_characters,self.batch_size,axis=-1), global_latent=samples, reuse=True, context_dim=self.decoder_units, max_time=self.num_sentence_characters)
	    predictions = self.decoder_p3(inputs=context,reuse=True,sequence_length=np.repeat(self.num_sentence_characters,self.batch_size,axis=-1),max_time=self.num_sentence_characters)	
            return predictions

    def vanilla_decoder(self,inputs,reuse):

        outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time, name='pred_char_array')

        cell = tf.contrib.rnn.LSTMCell(self.decoder_p3_units)

        def loop_fn(time, cell_output, cell_state, loop_state):
            next_loop_state = loop_state
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(self.batch_size, tf.float32)
                next_input = tf.concat(
                    [tf.zeros(shape=[self.batch_size, self.dict_length+self.lat_word_dim], dtype=tf.float32)],
                    axis=-1)
                next_loop_state = outputs_ta
            else:

                next_cell_state = cell_state
                prediction = tf.layers.dense(inputs=cell_output, activation=None, units=self.dict_length)
                next_loop_state = loop_state.write(time - 1, prediction)
                next_input = tf.concat([prediction, inputs], axis=-1)
            elements_finished = (time >= sequence_length - 1)

            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        with tf.variable_scope('vanilla_decoder', reuse=reuse):
            _, _, loop_ta = tf.nn.raw_rnn(cell, loop_fn)
            output = _transpose_batch_time(loop_ta.stack())
        return output


#Example usage
#batch_len = np.random.randint(low=0,high=30,size=[10])
#arg_dict = {'global_lat_dim':10,'word_lens':batch_len,'batch_size':10,'max_num_words':30,'decoder_units':40,'encodings' : np.random.randn(10,30,40),'sentence_lens':np.random.randint(low=0,high=30,size=10),'num_sentence_characters':200,'dict_length':26}
#decoder = Decoder(**arg_dict)
#word_encoding_placeholder=tf.placeholder(dtype=tf.float32,shape=[decoder.batch_size,decoder.max_num_words,np.shape(decoder.encodings)[-1]])

#out_o, global_latent_o,global_logsig_o,global_mu_o = decoder.run_decoder(units_lstm_decoder=40,lat_words=word_encoding_placeholder,units_dense_global=40,sequence_length=batch_len)
#true_mat =np.zeros(shape=[decoder.batch_size,decoder.num_sentence_characters],dtype=np.float32)
#for k,i in enumerate(batch_len):
#    true_mat[k,0:i] = np.random.randint(low=0,high=decoder.dict_length,size=[i])

#true_inp=true_mat

#posterior_mu =np.random.randn(10,30,40)
#posterior_logsig = np.exp(np.random.randn(10,30,40))

#cost= decoder.calc_cost(prior_mu=posterior_mu,prior_logsig=posterior_logsig,global_latent_sample=global_latent_o,global_logsig=global_logsig_o,global_mu=global_mu_o,predictions=out_o,true_input=tf.one_hot(indices=true_inp,depth =decoder.dict_length),posterior_logsig=posterior_logsig,posterior_mu=posterior_mu,post_samples=decoder.encodings)
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
 #   cost_o=sess.run([cost],feed_dict={word_encoding_placeholder:decoder.encodings})
