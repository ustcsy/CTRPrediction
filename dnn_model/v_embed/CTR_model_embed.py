#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np
sys.path.append('../../data_reader/')
import datareader_new as dr
sys.path.append('../../text_model/')
import text_model as t2v
from gensim.models import word2vec
from gensim.models import doc2vec
from sklearn.metrics import roc_auc_score

def load_data(data_type, data_path, data_length):
   	if data_type == 'dev':
		x_number = np.fromfile(data_path+'x_number.data')
        	x_label = np.fromfile(data_path+'x_label.data')
	        x_text_vec = np.fromfile(data_path+'x_text_vec.data')
        	y = np.fromfile(data_path+'y.data')
	        x_number.shape = data_length, 29
        	x_label.shape = data_length, 493
	        x_text_vec.shape = data_length, 200
        	y = y.astype(int)
        	y = np.eye(2)[y]
    	elif data_type == 'release':
        	[x_number, x_label, x_text, y] = dr.read_raw_data(data_path, data_length)
        	print '\tText2vector ...'
	        x_text_dim = 200
        	word2vec_model_path = '../text_model/merged_word_vec.model'
	        word2vec_model = word2vec.Word2Vec.load(word2vec_model_path)
        	x_text_vec = np.zeros(shape=(data_length, x_text_dim))
        	for i in range(data_length):
            		print i,'\r',
            		x_text_vec[i,:] = wv.sen2vec(x_text[i,0], word2vec_model, x_text_dim)
    	return [x_number, x_label, x_text_vec, y]

def batch_norm(x, node_num, bn_epsilon):
	batch_mean, batch_var = tf.nn.moments(x, [0])
	scale = tf.Variable(tf.ones([node_num]))
	beta = tf.Variable(tf.zeros([node_num]))
	return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, bn_epsilon)

def embed_layer(x_lab, weight1, weight2, bn_epsilon):
	e1 = tf.nn.relu(batch_norm(tf.matmul(x_lab, weight1), weight1.get_shape().as_list()[1], bn_epsilon))
	e2 = tf.nn.relu(batch_norm(tf.matmul(e1, weight2), weight2.get_shape().as_list()[1], bn_epsilon))
	return e2

def model(x_ph, weight, bias, bn_epsilon):
	h1_602 = embed_layer(x_ph['l602'], weight['eb602_1'], weight['eb602_2'], bn_epsilon)
	h1_603 = embed_layer(x_ph['l603'], weight['eb603_1'], weight['eb603_2'], bn_epsilon)
	h1_606 = embed_layer(x_ph['l606'], weight['eb606_1'], weight['eb606_2'], bn_epsilon)
	h1_607 = embed_layer(x_ph['l607'], weight['eb607_1'], weight['eb607_2'], bn_epsilon)
	h1_617 = embed_layer(x_ph['l617'], weight['eb617_1'], weight['eb617_2'], bn_epsilon)
	h1_619 = embed_layer(x_ph['l619'], weight['eb619_1'], weight['eb619_2'], bn_epsilon)
	h1_645 = embed_layer(x_ph['l645'], weight['eb645_1'], weight['eb645_2'], bn_epsilon)
	x_lab_tensor = tf.concat(1, [h1_602, h1_603, h1_606, h1_607, h1_617, h1_619, h1_645])

	h2_nontext = tf.nn.relu(batch_norm(tf.matmul(tf.concat(1, [x_ph['number'], x_lab_tensor]), weight['nontxt_1']), weight['nontxt_1'].get_shape().as_list()[1], bn_epsilon))
	h3_nontext = tf.nn.relu(batch_norm(tf.matmul(h2_nontext, weight['nontxt_2']), weight['nontxt_2'].get_shape().as_list()[1], bn_epsilon))
	h4 = tf.nn.relu(batch_norm(tf.matmul(tf.concat(1, [h3_nontext, x_ph['text']]), weight['fea_eng']), weight['fea_eng'].get_shape().as_list()[1], bn_epsilon))
	h5 = tf.nn.relu(batch_norm(tf.matmul(h4, weight['fea_sel1']), weight['fea_sel1'].get_shape().as_list()[1], bn_epsilon))
	h6 = tf.nn.relu(batch_norm(tf.matmul(h5, weight['fea_sel2']), weight['fea_sel2'].get_shape().as_list()[1], bn_epsilon))
	h7 = tf.nn.relu(batch_norm(tf.matmul(h6, weight['fea_sel3']), weight['fea_sel3'].get_shape().as_list()[1], bn_epsilon))
	h8 = tf.nn.relu(batch_norm(tf.matmul(h7, weight['tra1']), weight['tra1'].get_shape().as_list()[1], bn_epsilon))
	logits = tf.matmul(h8, weight['tra2'])+bias['tra2']
	y_ = tf.nn.softmax(logits)
	return [logits, y_]

def dnn_train(sess, train_op, loss, x_number, x_label, x_text_vec, y, x_ph, y_ph, data_length, batch_size, data_type):
	if data_type == 'dev':
		data_length = data_length/2
	index_arr = np.arange(data_length)
	np.random.shuffle(index_arr)
	start = 0
	while start < data_length:
		end = start + batch_size
		index_arr_tmp = index_arr[start:end]
		feed_dict={x_ph['number']: x_number[index_arr_tmp],
			   x_ph['l602']: x_label[index_arr_tmp, 0:46],
			   x_ph['l603']: x_label[index_arr_tmp, 46:189],
			   x_ph['l606']: x_label[index_arr_tmp, 189:319],
			   x_ph['l607']: x_label[index_arr_tmp, 319:352],
			   x_ph['l617']: x_label[index_arr_tmp, 352:355],
			   x_ph['l619']: x_label[index_arr_tmp, 355:375],
			   x_ph['l645']: x_label[index_arr_tmp, 375:493],
			   x_ph['text']: x_text_vec[index_arr_tmp], 
			   y_ph: y[index_arr_tmp]}
		_, loss_tmp = sess.run([train_op, loss], feed_dict=feed_dict)
		print loss_tmp, "\r",
		start = end
	if data_type == 'dev':
		dnn_test(sess, x_number[:data_length], x_label[:data_length], x_text_vec[:data_length], data_length, 'tmptrain.predicts')
		os.system("../../auc_calculate/fastAUC.bash tmptrain.predicts ../../../CTRData/1_data_y_train tmptrain.auc > tmptrain")
		auc_file_train = open('tmptrain.auc', 'r')
		train_auc = auc_file_train.readline().strip(' \t\r\n')
		dnn_test(sess, x_number[data_length:], x_label[data_length:], x_text_vec[data_length:], data_length, 'tmp.predicts')
		os.system("../../auc_calculate/fastAUC.bash tmp.predicts ../../../CTRData/1_data_y_test tmp.auc > tmp")
		auc_file_test = open('tmp.auc', 'r')
		test_auc = auc_file_test.readline().strip(' \t\r\n')
		print train_auc,'\t',test_auc,

def dnn_test(session, x_number, x_label, x_text, data_length, result_path):
	test_batch_size = 20000
	start = 0
	y_hat = np.arange(0)
	while start < data_length:
		end = start + test_batch_size
		x_num_tmp = x_number[start:end]
		x_lab_tmp = x_label[start:end]
		x_txt_tmp = x_text[start:end]
		feed_dict = {x_ph['number']:x_num_tmp,
			     x_ph['l602']: x_lab_tmp[:, 0:46],
			     x_ph['l603']: x_lab_tmp[:, 46:189],
			     x_ph['l606']: x_lab_tmp[:, 189:319],
			     x_ph['l607']: x_lab_tmp[:, 319:352],
			     x_ph['l617']: x_lab_tmp[:, 352:355],
			     x_ph['l619']: x_lab_tmp[:, 355:375],
			     x_ph['l645']: x_lab_tmp[:, 375:493],
			     x_ph['text']:x_txt_tmp}
		y_hat_tmp = session.run(y_, feed_dict=feed_dict)
		y_hat = np.hstack((y_hat, y_hat_tmp[:,1]))
		start = end
	np.savetxt(result_path, y_hat, fmt="%f", delimiter="\n")

if __name__=="__main__":
	if len(sys.argv) != 5:
		print 'Use: train/test data_file data_length model_file'
		exit(1)
	param = {
		'use_type': sys.argv[1],
		'file_path': sys.argv[2],
		'data_length': int(sys.argv[3]),
		'learn_type': 'dev',
		'batch_size': 1024,
		'step_size': 1e-5,
		'epoch_num': 100,
		'bn_epsilon': 1e-3,
		'device': '0'
	}
	os.environ["CUDA_VISIBLE_DEVICES"]=param['device']

	print 'Import data ...'
    	[x_number, x_label, x_text_vec, y] = load_data(param['learn_type'], param['file_path'], param['data_length'])
	
	# placeholder
	print "Initialize model ..."
	x_ph = {
		'number': tf.placeholder(tf.float32, [None, 29]),
		'l602': tf.placeholder(tf.float32, [None, 46]),
		'l603': tf.placeholder(tf.float32, [None, 143]),
		'l606': tf.placeholder(tf.float32, [None, 130]),
		'l607': tf.placeholder(tf.float32, [None, 33]),
		'l617': tf.placeholder(tf.float32, [None, 3]),
		'l619': tf.placeholder(tf.float32, [None, 20]),
		'l645': tf.placeholder(tf.float32, [None, 118]),
		'text': tf.placeholder(tf.float32, [None, 200])
	}
	y_ph = tf.placeholder(tf.float32, [None, 2])

	weight = {
		'eb602_1': tf.Variable(tf.random_normal([46, 46], stddev=np.sqrt(2./46/46))),
		'eb603_1': tf.Variable(tf.random_normal([143, 143], stddev=np.sqrt(2./143/143))),
		'eb606_1': tf.Variable(tf.random_normal([130, 130], stddev=np.sqrt(2./130/130))),
		'eb607_1': tf.Variable(tf.random_normal([33, 33], stddev=np.sqrt(2./33/33))),
		'eb617_1': tf.Variable(tf.random_normal([3, 3], stddev=np.sqrt(2./3/3))),
		'eb619_1': tf.Variable(tf.random_normal([20, 20], stddev=np.sqrt(2./20/20))),
		'eb645_1': tf.Variable(tf.random_normal([118, 118], stddev=np.sqrt(2./118/118))),
		
		'eb602_2': tf.Variable(tf.random_normal([46, 4], stddev=np.sqrt(2./46/4))),
		'eb603_2': tf.Variable(tf.random_normal([143, 4], stddev=np.sqrt(2./143/4))),
		'eb606_2': tf.Variable(tf.random_normal([130, 4], stddev=np.sqrt(2./130/4))),
		'eb607_2': tf.Variable(tf.random_normal([33, 4], stddev=np.sqrt(2./33/4))),
		'eb617_2': tf.Variable(tf.random_normal([3, 4], stddev=np.sqrt(2./3/4))),
		'eb619_2': tf.Variable(tf.random_normal([20, 4], stddev=np.sqrt(2./20/4))),
		'eb645_2': tf.Variable(tf.random_normal([118, 4], stddev=np.sqrt(2./118/4))),

		'nontxt_1': tf.Variable(tf.random_normal([57, 256], stddev=np.sqrt(2./57/256))),
		'nontxt_2': tf.Variable(tf.random_normal([256, 512], stddev=np.sqrt(2./256/512))),
		'fea_eng': tf.Variable(tf.random_normal([712, 1500], stddev=np.sqrt(2./712/1500))),
		'fea_sel1': tf.Variable(tf.random_normal([1500, 300], stddev=np.sqrt(2./1500/300))),
		'fea_sel2': tf.Variable(tf.random_normal([300, 100], stddev=np.sqrt(2./300/100))),
		'fea_sel3': tf.Variable(tf.random_normal([100, 50], stddev=np.sqrt(2./100/50))),
		'tra1': tf.Variable(tf.random_normal([50, 20], stddev=np.sqrt(2./50/20))),
		'tra2': tf.Variable(tf.random_normal([20, 2], stddev=np.sqrt(2./20/2)))
	}

	bias = {
		'tra2': tf.Variable(tf.constant(0.001, shape=[2]))
	}

	[logits, y_] = model(x_ph, weight, bias, param['bn_epsilon'])

	# initialization
	print "Initialize session ..."
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_ph))
	train_op = tf.train.AdamOptimizer(learning_rate=param['step_size']).minimize(loss)

	# session open
	saver = tf.train.Saver()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:	
		sess.run(tf.initialize_all_variables())
		if param['use_type'] == 'train':
			print "Model train ..."
			for i in range(param['epoch_num']):
				dnn_train(sess, train_op, loss, x_number, x_label, x_text_vec, y, x_ph, y_ph, param['data_length'], param['batch_size'], param['learn_type'])
			save_path = saver.save(sess, sys.argv[4])
			print "Model saved in file ..."
		elif param['use_type'] == 'test':
			saver.restore(sess, sys.argv[4])
			print "Model restored for testing ..."
			dnn_test(sess, x_number, x_label, x_text_vec, param['data_length'], 'ctr.predicts')
		print "Prediction result saved ..."
