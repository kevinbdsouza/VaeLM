#unrolling LSTM example


    def decoder_three_generation_sample(self, input_encoder, reuse=True):
        # Multi RNN with proper teacher forcing
        #with tf.variable_scope('decoder_three', reuse=reuse):
            #l1 = tf.contrib.layers.batch_norm(input_encoder, is_training=False)
            #l1=tf.layers.dense(input_encoder, units=self.lat_dim,activation=tf.nn.relu)
        y0 = tf.zeros([self.batch_size, self.char_len], dtype=tf.float32)
        y_z = tf.concat([y0, input_encoder], axis=-1)
        h2 = tf.zeros([self.batch_size, 1, self.char_len], dtype=tf.float32)
        i0 = tf.constant(0)
        # might have to make this lstm tuple, initialization is meaningless, will change in loop
        s0_shape = tf.contrib.rnn.LSTMStateTuple(tf.TensorShape([None, 1024]),tf.TensorShape([None, 1024]))
        s0 = tf.contrib.rnn.LSTMStateTuple(tf.zeros([self.batch_size,1024]), tf.zeros([self.batch_size, 1024], dtype=tf.float32))
        time_steps = self.mol_len

        def c(i, s0, h2, y_z):
            return i < time_steps

        def b1(i, s0, h2, y_z):
            with tf.variable_scope("decoder_three", reuse=True):
                with tf.variable_scope("rnn"):
                    cell = tf.contrib.rnn.LSTMCell(1024)
                    if i == 0:
                        s0 = cell.zero_state(self.batch_size, dtype=tf.float32)
                    y_z = tf.reshape(y_z, shape=[self.batch_size, (self.char_len + self.lat_dim)])
                    outputs_1, state = cell(y_z, s0)
                outputs_2 = tf.layers.dense(outputs_1, units=self.char_len, activation=None, name="final_dense")
                    # greedy Decoding
                samples = tf.multinomial(outputs_2,1,seed=2007)
                outputs = tf.one_hot(samples, depth=self.char_len, axis=-1, dtype=tf.float32)
            return [i + 1, state, tf.concat([h2, tf.reshape(outputs, [self.batch_size, 1, self.char_len])], axis=1),
                    tf.concat([tf.reshape(outputs,[self.batch_size,self.char_len]), input_encoder], axis=1)]

        # [tf.TensorShape(self.batch_size, None, self.char_len),tf.TensorShape(self.batch_size, self.lat_dim + self.char_len)]
        ii, s0, h2, y_z = tf.while_loop(c, b1, loop_vars=[i0, s0, h2, y_z], shape_invariants=[i0.get_shape(), s0_shape,
                                                                                              tf.TensorShape(
                                                                                                  [self.batch_size, None,
                                                                                                   self.char_len]),
                                                                                              y_z.get_shape()])
        h2 = tf.slice(h2, [0, 1, 0], [-1, -1, -1])
        return h2
