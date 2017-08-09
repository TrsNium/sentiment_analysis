import tensorflow as tf
import argparse
from util import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random

class model():
    def __init__(self):
        #self.args = args

        #Character lever CNN RNN
        max_word_length = 62
        vocab_size = 45
        embedding_size = 650
        max_time_step = 100
        batch_size = 8
        cell_model = "lstm"
        num_layers = 1
        rnn_size = 4096
        filter_nums = [32,62,128,256,256,256]
        kernels = [2,3,4,5,6,7]
        
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, max_time_step, max_word_length])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        
        with tf.variable_scope('Embedding_and_Conv') as scope:
            cnn_outputs = []
            
            embedding_weight = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1.,1.), name='embedding_weight')
            word_t  = tf.split(self.inputs, max_time_step, axis=1)
            for t in range(max_time_step):
                if t is not 0:
                    tf.get_variable_scope().reuse_variables()

                char_index = tf.reshape(word_t[t], shape=[-1, max_word_length])
                embedded = tf.nn.embedding_lookup(embedding_weight, char_index)
                embedded_ = tf.expand_dims(embedded, -1)

                t_cnn_outputs = []
                for kernel, filter_num in zip(kernels, filter_nums):
                    conv_ = tf.layers.conv2d(embedded_, filter_num, kernel_size=[kernel, embedding_size], padding="valid", strides=[1, 1], activation=tf.nn.relu, name="conv_{}".format(kernel))
                    pool_ = tf.layers.max_pooling2d(conv_, pool_size=[vocab_size-kernel+1, 1], strides=[vocab_size-kernel+1, 1])
                    t_cnn_outputs.append(tf.reshape(pool_, (-1, filter_num)))

                #print(tf.convert_to_tensor(t_cnn_outputs).get_shape().as_list())
                cnn_output = tf.concat([t_cnn_output for t_cnn_output in t_cnn_outputs], axis=-1)
                cnn_outputs.append(self.highway(cnn_output, sum(filter_nums), tf.nn.sigmoid))

            cnn_outputs = tf.convert_to_tensor(cnn_outputs)
            
        if cell_model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif cell_model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif cell_model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(self.args.cell_model))

        with tf.variable_scope('RNN') as scope:
            def cell():
                return cell_fn(rnn_size, reuse=tf.get_variable_scope().reuse)
        
            cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(num_layers)], state_is_tuple = True)
            state_in = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            rnn_out, self.out_state = tf.nn.dynamic_rnn(cell, cnn_outputs, initial_state=state_in, time_major=True,dtype=tf.float32)
        
        with tf.variable_scope("dense_layer") as scope:
            logits = tf.layers.dense(rnn_out[-1], 2, name="Dense")
            self.outs = tf.nn.softmax(logits)

        with tf.variable_scope("loss") as scope:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
    
    def highway(self, x, size, activation, carry_bias=-1.0):
        T = tf.layers.dense(x, size, activation=tf.nn.sigmoid, name="transfort_gate")
        H = tf.layers.dense(x, size, activation=activation, name="activation")
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y
    
    def train(self):
        opt_ = tf.train.AdamOptimizer(0.002).minimize(self.loss)
        
        batch_size = 8
        labels, train = mk_char_level_cnn_rnn_train_data("data/train.txt", "data/char_index.txt", 100, 62)
        train_inp, test_inp, train_labels, test_labels = train_test_split(train, labels, test_size=0.33, random_state=42)
        train_data_size = train_inp.shape[0]
        #print(train_inp.shape, test_inp.shape, train_labels.shape, test_labels.shape, train_data_size)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter("./logs", sess.graph)
            
            for itr in range(100000):
                choiced_idx = random.sample(range(train_data_size), batch_size)
                loss, _ = sess.run([self.loss, opt_], feed_dict={self.inputs: train_inp[choiced_idx], self.labels:train_labels[choiced_idx]})

                if itr % 10 == 0:
                    labels = train_labels[choiced_idx]
                    choiced_idx = random.sample(range(train_data_size), batch_size)
                    loss, out = sess.run([self.loss, self.outs], feed_dict={self.inputs: train_inp[choiced_idx], self.labels:labels})
                    accuracy = len([i for i in range(batch_size) if np.argmax(labels[i], axis=-1) == np.argmax(out[i], axis=-1)])/batch_size
                    print("itr:",itr,"    loss:", loss, out, accuracy)
            
                if itr % 1000 == 0:
                    saver.save(sess, 'saved/model.ckpt', itr)
                    print('-----------------------saved model-------------------------')