#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import re
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

# read raw data
def read_raw_data( file_path, data_length ):
	x_num_index = [15, 19, 21, 25, 27, 29, 31, 33, 35, 37, 39, 43, 49, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 77, 79, 81, 87, 89, 91]
	# 602 [channelFstLevel] : [0,45]	len[46]
	# 605 [titleIsImageSet] : [0,1] 	len[1]
	# 617 [userSex] : [0,2]			len[3]
	# 619 [showListLoc]: [1,20]		len[20]
	# 620 [crossAreaNewsArea] : [0,1]	len[1]
	# 648 [newsIsListImage] : [0,1]		len[1]
	# 652 [newsIsBigImage] : [0,1]		len[1]
	x_lab_index = [17, 23, 47, 51, 53, 85, 93]
	# 502 [titleText] : [string length=100]
	x_text_index= [5]
	y_index = [1]

	x_num = np.zeros(shape=(data_length, len(x_num_index)))
	x_lab = np.zeros(shape=(data_length, 73))
	x_text = np.full(shape=(data_length, 1), fill_value="", dtype='|S100')
	y = np.zeros(shape=(data_length, 1))	

	data_file = open(file_path, 'r')
	title_text = open("title_corpus_new", 'w')
	line_cnt = 0
	for line in data_file.readlines():
		#line = line.decode('gbk').encode('utf8')
		line = line.decode('gbk')
		line = line.strip()
		arr_tmp = line.split()
		if len(arr_tmp) == 94:
			x_num_tmp = [arr_tmp[i] for i in x_num_index]
			x_lab_tmp = [arr_tmp[i] for i in x_lab_index]
			x_text_tmp = [arr_tmp[i] for i in x_text_index]
			x_text_tmp = x_text_tmp[0]
			y_tmp = [arr_tmp[i] for i in y_index]
			y_tmp = y_tmp[0]
		else:
			regStr = '''^(?P<a0>.*)\t(?P<a1>.*)\t500\t(?P<a3>.*)\t501\t(?P<a5>.*)\t502\t(?P<a7>.*)\t503\t(?P<a9>.*)\t504\t(?P<a11>.*)\t505\t(?P<a13>.*)\t601\t(?P<a15>.*)\t602\t(?P<a17>.*)\t603\t(?P<a19>.*)\t604\t(?P<a21>.*)\t605\t(?P<a23>.*)\t606\t(?P<a25>.*)\t607\t(?P<a27>.*)\t608\t(?P<a29>.*)\t609\t(?P<a31>.*)\t610\t(?P<a33>.*)\t611\t(?P<a35>.*)\t612\t(?P<a37>.*)\t613\t(?P<a39>.*)\t614\t(?P<a41>.*)\t615\t(?P<a43>.*)\t616\t(?P<a45>.*)\t617\t(?P<a47>.*)\t618\t(?P<a49>.*)\t619\t(?P<a51>.*)\t620\t(?P<a53>.*)\t621\t(?P<a55>.*)\t622\t(?P<a57>.*)\t623\t(?P<a59>.*)\t624\t(?P<a61>.*)\t625\t(?P<a63>.*)\t626\t(?P<a65>.*)\t627\t(?P<a67>.*)\t628\t(?P<a69>.*)\t629\t(?P<a71>.*)\t630\t(?P<a73>.*)\t631\t(?P<a75>.*)\t632\t(?P<a77>.*)\t633\t(?P<a79>.*)\t645\t(?P<a81>.*)\t647\t(?P<a83>.*)\t648\t(?P<a85>.*)\t649\t(?P<a87>.*)\t650\t(?P<a89>.*)\t651\t(?P<a91>.*)\t652\t(?P<a93>.*)$'''
			reg = re.compile(regStr)
        		regMatch = reg.match(line)
        		linebits = regMatch.groupdict()
			x_num_tmp = []
			for i in x_num_index:
				x_num_tmp.append(linebits['a'+str(i)])
			x_lab_tmp = []
			for i in x_lab_index:
				x_lab_tmp.append(linebits['a'+str(i)])
			x_text_tmp = linebits['a5']
			y_tmp = linebits['a1']

		x_num_tmp = np.asarray(x_num_tmp).astype("float")
		x_lab_tmp = np.asarray(x_lab_tmp).astype("int")
		y_tmp = np.asarray(y_tmp).astype("float")
		x_text_tmp = x_text_tmp.split('\x03')
		x_text_tmp = ' '.join(x_text_tmp)
		x_text_tmp = x_text_tmp.encode('utf8')
		#print "{}: {}".format(line_cnt, x_text_tmp)
		print line_cnt,"\r",
		
		title_text.write(x_text_tmp+'\n')
		
		x_num[line_cnt, :] = x_num_tmp
		x_lab[line_cnt, 0:46] = np.eye(46)[x_lab_tmp[0]]
		x_lab[line_cnt, 46]   = x_lab_tmp[1]
		x_lab[line_cnt, 47:50] = np.eye(3)[x_lab_tmp[2]]
		x_lab[line_cnt, 50:70] = np.eye(20)[x_lab_tmp[3]-1]
		x_lab[line_cnt, 70] = x_lab_tmp[4]
		x_lab[line_cnt, 71] = x_lab_tmp[5]
		x_lab[line_cnt, 72] = x_lab_tmp[6]
		x_text[line_cnt] = x_text_tmp
		y[line_cnt] = y_tmp
		line_cnt = line_cnt + 1
		print line_cnt,'\r',

	mean_number = [2.76700500996e+13, 8.51185175, 737.509099375, 64.54679825, 12.10352575, 1.09050325, 3.254765, 0.189887344247, 0.0182185186881, 0.582897932147, 58525.1128065, 0.00323515595172, 2.1508435, 0.109486821681, 0.100562446046, 0.100374183902, 0.112807965744, 53497.4601047, 6633027.30965, 2806267.01902, 140148.424155, 0.00474227594024, 0.00390079259991, 0.0060802298141, 0.00474841301374, 39.135987375, 13095.282943, -0.008962625, 0.14179575 ]
	std_number = [2.25924842747e+16, 7.51826501169, 909.783735654, 41.5310178502, 2.79110523254, 0.565585017252, 0.958294993608, 0.167664388836, 0.0192172023312, 0.252080772164, 160188.849809, 0.00874981007977, 19.4899225239, 0.0760034257447, 0.0244761195675, 0.0249677456445, 0.0494568517436, 134435.624751, 4479273.75764, 2594456.42461, 249033.090384, 0.0133638473941, 0.0118200778527, 0.0123480903815, 0.0111911306576, 7.57338507927, 101498.458489, 1.38044523483, 1.77176013198]
	for i in range(29):
		x_num[:, i] = (x_num[:, i] - mean_number[i]) / std_number[i]
	return x_num, x_lab, x_text, y
