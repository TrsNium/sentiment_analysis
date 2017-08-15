import sys
sys.path.append("../")

import tensorflow as tf
from util import *
import numpy as np
import os
import argparser

class model():
    def __init__(self, args):
        self.args = args

        self.inputs = tf.placeholder(dtype=tf.float32, [None, args.max_time_step, args.vocab_size], name="inputs")
        self.labels = tf.placeholder(dtype=tf.float32, [None, args.max_time_step, 2])

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
                cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=args.rnn_keep_prob)
            return cell_

        with tf.variable_scope("RNN"):
            cell_ = tf.contrib.rnn.MultiRNNCell([def_cell() for _ in range(args.num_layers)], state_is_tuple = True)
            rnn_outs, final_state = tf.nn.dynamic_rnn(cell_, self.inputs, initial_state=cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32), dtype=tf.float32, name="RNN")
        
        with tf.variable_scope("dense"):
            reshaped_rnn_outs = tf.transpose(rnn_outs, (1,0,2))
            if args.merge_all_timestep:
                reduced = tf.reduce_sum(reshaped_rnn_outs, axis=0)
                logits = tf.layers.dense(reduced, 2, activation=tf.nn.sigmoid, name="final_outputs_dense")
            else:
                logits = tf.layers.dense(reshaped_rnn_outs[-1], 2, activation=tf.nn.sigmoid, name="final_outputs_dense")

            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            self.outs = tf.nn.softmax(logits)
        
    def train(self):
        opt_ = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            graph = tf.summary.FileWriter('./logdir/', graph = sess.graph)
            projector.visualize_embeddings(graph, embedding_config)
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())

            for itr in self.args.itrs:

                if itr % 20 == 0:
                    pass

                if itr % 1000 == 0:
                    saver.save(sess, "save/model.ckpt")
                    print("--------------------saved model-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.02)
    parser.add_argument("--cell_model", dest="cell_model", type= str, default="gru")
    parser.add_argument("--data_dir", dest="data_dir", type=str, default="../data/")
    parser.add_argument("--num_layers", dest="num_layers", type=int, default=1)
    parser.add_argument("--rnn_size", dest="rnn_size", type=int, default=512)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=40)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2348)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    args = parser..parse_args()

    model_ = model(args)
    if args.train:
        model_.train()

