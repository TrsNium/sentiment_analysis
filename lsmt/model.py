import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from util import *
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random

class model():
    def __init__(self, args):
        self.args = args

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, args.vocab_size], name="inputs")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        if args.cell_type == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.cell_type == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif args.cell_type == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.cell_type))

        def def_cell():
            cell_ = cell_fn(args.rnn_size, reuse=tf.get_variable_scope().reuse)
            if args.keep_prob < 1.:
                cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=args.keep_prob)
            return cell_

        with tf.variable_scope("RNN"):
            cell_ = tf.contrib.rnn.MultiRNNCell([def_cell() for _ in range(args.num_layers)], state_is_tuple = True)
            rnn_outs, final_state = tf.nn.dynamic_rnn(cell_, self.inputs, initial_state=cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32), dtype=tf.float32)
        
        with tf.variable_scope("dense"):
            reshaped_rnn_outs = tf.transpose(rnn_outs, (1,0,2))
            if args.merge_all_timestep:
                reduced = tf.reduce_sum(reshaped_rnn_outs, axis=0)
                logits = tf.layers.dense(reduced, 2, activation=tf.nn.sigmoid, name="final_outputs_dense")
            else:
                logits = tf.layers.dense(reshaped_rnn_outs[-1], 2, activation=tf.nn.sigmoid, name="final_outputs_dense")

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
            self.out = tf.nn.softmax(logits)
        
    def train(self):
        opt_ = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.loss)
        
        train_labels, train_inp = mk_train_onehot_data("../data/train.txt", "../data/index.txt", self.args.max_time_step)
        if self.args.test:
            train_inp, test_inp, train_labels, test_labels = train_test_split(train_inp, train_labels, test_size=0.33, random_state=42)
            test_data_size = test_inp.shape[0]
        train_data_size = train_inp.shape[0]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            graph = tf.summary.FileWriter('./logdir/', graph = sess.graph)
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())

            for itr in range(self.args.itrs):
                choiced_idx = random.sample(range(train_data_size), self.args.batch_size)
                loss, _ = sess.run([self.loss, opt_], feed_dict={self.inputs: train_inp[choiced_idx], self.labels:train_labels[choiced_idx]})
                if itr % 20 == 0:
                    choiced_idx = random.sample(range(train_data_size), self.args.batch_size)
                    labels = train_labels[choiced_idx]
                    loss, out = sess.run([self.loss, self.out], feed_dict={self.inputs: train_inp[choiced_idx], self.labels:labels})
                    acctualy = len([i for i in range(self.args.batch_size) if np.argmax(out, -1)[i] == np.argmax(labels, -1)[i]])/self.args.batch_size
                    print("itr:",itr,"    loss:", loss, acctualy)

                if itr % 1000 == 0:
                    saver.save(sess, "save/model.ckpt")
                    print("--------------------saved model-------------------")

            if self.args.test:
                acctualy_ = 0.                
                for i in range(int(test_data_size/self.args.batch_size)):
                    labels = test_labels[i*self.args.batch_size:(i+1)*self.args.batch_size]
                    out = sess.run(self.out, feed_dict={self.inputs: test_inp[i*self.args.batch_size:(i+1)*self.args.batch_size]})
                    acctualy = len([i for i in range(self.args.batch_size) if np.argmax(out, -1)[i] == np.argmax(labels, -1)[i]])/self.args.batch_size
                    acctualy_ += acctualy
                    print(acctualy)
                print("avg", acctualy_/(int(test_data_size/self.args.batch_size)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--merge_all_timestep", dest="merge_all_timestep", type=bool, default=False)
    parser.add_argument("--lr", dest="lr", type=float, default= 0.02)
    parser.add_argument("--cell_type", dest="cell_type", type= str, default="lstm")
    parser.add_argument("--data_dir", dest="data_dir", type=str, default="../data/")
    parser.add_argument("--num_layers", dest="num_layers", type=int, default=2)
    parser.add_argument("--rnn_size", dest="rnn_size", type=int, default=256)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2347)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--test", dest="test", type=bool, default=True)
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=0.4)
    parser.add_argument("--betch_size", dest="batch_size", type=int, default=20)
    parser.add_argument("--itrs", dest="itrs", type=int, default=7001) 
    args = parser.parse_args()

    if not os.path.exists("save"):
        os.mkdir("save")

    if not os.path.exists("logs"):
        os.mkdir("logs")


    model_ = model(args)
    if args.train:
        model_.train()

