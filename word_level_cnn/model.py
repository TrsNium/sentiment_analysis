import tensorflow as tf
import argparse
from util import *
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random


class model():
    def __init__(self):
        self.args =  args
        
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, args.max_time_step, 1])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        
        with tf.variable_scope("Embedding") as scope:
            splitted_word_ids  = tf.split(self.inputs, args.max_time_step, axis=1)
            embedding_weight = tf.Variable(tf.random_uniform([args.vocab_size, args.embedding_size],-1.,1.), name='embedding_weight')
            t_embedded = []
            
            for t in range(args.max_time_step):
                if t is not 0:
                    tf.get_variable_scope().reuse_variables()

                embedded = tf.nn.embedding_lookup(embedding_weight, self.inputs[:,t,:])
                t_embedded.append(embedded)
            reshaped_embedded = tf.reshape(tf.transpose(tf.convert_to_tensor(t_embedded), perm=(1,0,2,3)), (-1, args.max_time_step, args.embedding_size))
            cnn_inputs = tf.expand_dims(reshaped_embedded, axis=-1)
            
        kernels = [1,2,3,5]
        filter_nums = [32,64,128,128]
        with tf.variable_scope("CNN") as scope:
            convded = []
            for kernel in kernels:
                conv_ = tf.layers.conv2d(cnn_inputs, filter_num, kernel_size=[kernel, args.embedding_size], strides=[1, 1], activation=tf.nn.relu, name="conv_{}".format(kernel))
                pool_ = tf.layers.max_pooling2d(conv_, pool_size=[args.max_time_step-kernel+1, 1], strides=[1, 1])
                convded.append(tf.reshape(pool_, (-1, filter_num)))
            convded = tf.concat([cnn_output for cnn_output in convded], axis=-1)
        
        with tf.variable_scope("Dense") as scope:
            flatten_ = tf.contrib.layers.flatten(convded)
            logits = tf.layers.dense(flatten_, 2, name="dense_layer")
            self.out = tf.nn.softmax(logits)
            
        with tf.variable_scope("loss") as scope:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        
        with tf.variable_scope("summary") as scope:
            tf.summary.scalar("loss", self.loss)

    def train(self):
        batch_size = 10
        opt_ = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)
        
        labels, train = mk_train_data("data/train.txt", "data/index.txt", self.args.max_time_step)
        train_inp, test_inp, train_labels, test_labels = train_test_split(train, labels, test_size=0.33, random_state=42)
        train_data_size = train_inp.shape[0]
        print(train_inp.shape, test_inp.shape, train_labels.shape, test_labels.shape, train_data_size)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            summary = tf.summary.merge_all()
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter("./logs", sess.graph)
            
            for itr in range(100000):
                choiced_idx = random.sample(range(train_data_size), batch_size)
                loss, _ = sess.run([self.loss, opt_], feed_dict={self.inputs: train_inp[choiced_idx], self.labels:train_labels[choiced_idx]})

                if itr % 100 == 0:
                    choiced_idx = random.sample(range(train_data_size), batch_size)
                    loss, merged_summary, out = sess.run([self.loss, summary, self.out], feed_dict={self.inputs: train_inp[choiced_idx], self.labels:train_labels[choiced_idx]})
                    graph.add_summary(merged_summary, itr)
                    print("itr:",itr,"    loss:", loss, out)
            
                if itr % 1000 == 0:
                    saver.save(sess, 'saved/model.ckpt', itr)
                    print('-----------------------saved model-------------------------')
