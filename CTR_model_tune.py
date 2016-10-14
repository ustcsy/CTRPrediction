#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np
import datareader as dr
import text_model as t2v
from gensim.models import word2vec
from gensim.models import doc2vec
from sklearn.metrics import roc_auc_score

# initial weight
def init_weights(shape):
	initial = tf.random_normal(shape, stddev=0.001)
	return tf.Variable(initial)

def init_bias(shape):
	initial = tf.constant(0.001, shape=shape)
	return tf.Variable(initial)

def dnn_test(session, x_number, x_label, x_text, data_length):
	test_batch_size = 20000
	start = 0
	y_hat = np.arange(0)
	while start < data_length:
		end = start + test_batch_size
		x_num_tmp = x_number[start:end]
		x_lab_tmp = x_label[start:end]
		x_txt_tmp = x_text[start:end]
		feed_dict = {x_number_ph:x_num_tmp, x_label_ph:x_lab_tmp, x_text_ph:x_txt_tmp}
		y_hat_tmp = session.run(y_, feed_dict=feed_dict)
		y_hat = np.hstack((y_hat, y_hat_tmp[:,1]))
		start = end
	np.savetxt("tmp.predicts", y_hat, fmt="%f", delimiter="\n")

if __name__=="__main__":
	if len(sys.argv) != 5:
		print 'Use: train/test data_file data_length model_file'
		exit(1)

	# get parameter
	use_type = sys.argv[1]
	file_path = sys.argv[2]
	data_length = int(sys.argv[3])
	# import data [ x_number, x_label, x_text, y ]
	print "Import data ..."
	#[x_number, x_label, x_text, y] = dr.read_raw_data(file_path, data_length)
	x_number = np.fromfile("../x_number.data")
	x_number.shape = 8000000, 30
	x_number_train = x_number[0:4000000, :]
	x_number_test  = x_number[4000000:, :]
	x_label = np.fromfile("../x_label.data")
	x_label.shape = 8000000, 6
	x_label_train = x_label[0:4000000, :]
	x_label_test  = x_label[4000000:, :]
	#x_text = np.full(shape=(8000000,1),fill_value="",dtype='|S100')
	#x_text_file = open("../title_corpus", 'r')
	#line_cnt = 0
	#for line in x_text_file.readlines():
	#	x_text[line_cnt, 0] = line.strip()
	#	line_cnt = line_cnt + 1
	y = np.fromfile("../y.data")
	y = y.astype(int)
	y = np.eye(2)[y]
	y_train = y[0:4000000]
	y_test  = y[4000000:]
	x_number_dim = x_number.shape[1]
	x_label_dim = x_label.shape[1]
	x_text_dim = 200
	batch_size = 200

	# text 2 vec
	print "Text2vector ..."
	#word2vec_model_path = "text_model/gens_doc2vec/gd2v_sen2vec.model"
	#word2vec_model = doc2vec.Doc2Vec.load(word2vec_model_path)
	#x_text_vec = np.zeros(shape=(data_length, x_text_dim))
	#for i in range(data_length):
	#	print i,'\r',
	#	x_text_vec[i,:] = t2v.gd2v_sen2vec(x_text[i,0], word2vec_model, x_text_dim)
	#x_text_vec.tofile("../x_text_vec_s.data")
	x_text_vec = np.fromfile("../x_text_vec_s.data")
	x_text_vec.shape = 8000000, 200
	x_text_vec_train = x_text_vec[0:4000000,:]
	x_text_vec_test  = x_text_vec[4000000:]
	
	# placeholder
	print "Initialize model ..."
	x_number_ph = tf.placeholder(tf.float32, [None, x_number_dim])
	x_label_ph = tf.placeholder(tf.float32, [None, x_label_dim])
	x_text_ph = tf.placeholder(tf.float32, [None, x_text_dim])
	y_ph = tf.placeholder(tf.float32, [None, 2])
	# weights notext
	w_num = init_weights([30, 30])
	b_num = init_bias([30])
	h1_num = tf.nn.relu(tf.matmul(x_number_ph, w_num)+b_num)
	w_lab = init_weights([6, 6])
	b_lab = init_bias([6])
	h1_lab = tf.nn.relu(tf.matmul(x_label_ph, w_lab)+b_lab)
	w_nontxt = init_weights([36, 36])
	b_nontxt = init_bias([36])
	h2_nontext = tf.nn.relu(tf.matmul(tf.concat(1, [h1_num, h1_lab]), w_nontxt)+b_nontxt)
	# text2vec
	# feture engineer all
	w_fea_eng = init_weights([236, 500])
	b_fea_eng = init_bias([500])
	h3 = tf.nn.relu(tf.matmul(tf.concat(1, [h2_nontext, x_text_ph]), w_fea_eng)+b_fea_eng)
	# feature selection
	w_fea_sel1 = init_weights([500, 300])
	b_fea_sel1 = init_bias([300])
	h4 = tf.nn.relu(tf.matmul(h3, w_fea_sel1)+b_fea_sel1)
	w_fea_sel2 = init_weights([300, 100])
	b_fea_sel2 = init_bias([100])
	h5 = tf.nn.relu(tf.matmul(h4, w_fea_sel2)+b_fea_sel2)
	w_fea_sel3 = init_weights([100, 50])
	b_fea_sel3 = init_bias([50])
	h6 = tf.nn.relu(tf.matmul(h5, w_fea_sel3)+b_fea_sel3)
	# model train
	w_tra1 = init_weights([50, 20])
	b_tra1 = init_bias([20])
	h7 = tf.nn.relu(tf.matmul(h6, w_tra1)+b_tra1)
	w_tra2 = init_weights([20, 2])
	b_tra2 = init_bias([2])
	logits = tf.matmul(h7, w_tra2)+b_tra2
	y_ = tf.nn.softmax(logits)

	# dropout
	p_keep_input = tf.placeholder(tf.float32)
	p_keep_hidden= tf.placeholder(tf.float32)
	
	# initialization
	print "Initialize session ..."
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_ph))
	#train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)

	# session open
	saver = tf.train.Saver()
	with tf.Session() as sess:	
		sess.run(tf.initialize_all_variables())
		if use_type == 'train':
			print "Model train ..."
			for i in range(100):
				#print "epoch ", i
				index_arr = np.arange(data_length/2)
				np.random.shuffle(index_arr)
				start = 0
				while start < data_length/2:
					end = start + batch_size
					index_arr_tmp = (index_arr[start:end], )
					#print x_number[index_arr_tmp]
					feed_dict={x_number_ph: x_number_train[index_arr_tmp], 
						   x_label_ph: x_label_train[index_arr_tmp], 
						   x_text_ph: x_text_vec_train[index_arr_tmp], 
						   y_ph: y_train[index_arr_tmp]}
					_, loss_tmp = sess.run([train_op, loss], feed_dict=feed_dict)
					print loss_tmp, "\r",
					start = end
				dnn_test(sess, x_number_test, x_label_test, x_text_vec_test, data_length/2)
				os.system("./fastAUC.bash tmp.predicts ../1_data_y_test tmp.auc > tmp")
				auc_file = open('tmp.auc', 'r')
				train_auc = auc_file.readline()
				print train_auc,
			save_path = saver.save(sess, sys.argv[4])
			print "Model saved in file ..."
		elif use_type == 'test':
			saver.restore(sess, sys.argv[4])
			print "Model restored for testing ..."
			result_file = open("CTRPrediction_adam.preds", 'w')
			start = 0
			for i in range(data_length):
				x_num_tmp = x_number[i+start, :]
				x_num_tmp.shape = 1, x_number_dim
				x_lab_tmp = x_label[i+start, :]
				x_lab_tmp.shape = 1, x_label_dim
				x_txt_tmp = x_text_vec[i+start, :]
				x_txt_tmp.shape = 1, x_text_dim
				feed_dict={x_number_ph: x_num_tmp, 
					   x_label_ph: x_lab_tmp, 
					   x_text_ph: x_txt_tmp}
				y_hat = sess.run(y_, feed_dict=feed_dict)
				print i,'\r',
				result_file.write(str(y_hat[0][1])+'\n')
			print "Prediction result saved ..."
