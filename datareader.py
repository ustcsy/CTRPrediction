#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import re
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

# read raw data
def read_raw_data( file_path, data_length ):
	x_num_index = [15, 19, 21, 25, 27, 29, 31, 33, 35, 37, 39, 43, 49, 51, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 77, 79, 81, 87, 89, 91]
	x_lab_index = [17, 23, 47, 53, 85, 93]
	x_text_index= [5]
	y_index = [1]

	x_num = np.zeros(shape=(data_length, len(x_num_index)))
	x_lab = np.zeros(shape=(data_length, len(x_lab_index)))
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
		x_lab_tmp = np.asarray(x_lab_tmp).astype("float")
		y_tmp = np.asarray(y_tmp).astype("float")
		x_text_tmp = x_text_tmp.split('\x03')
		x_text_tmp = ' '.join(x_text_tmp)
		x_text_tmp = x_text_tmp.encode('utf8')
		#print "{}: {}".format(line_cnt, x_text_tmp)
		print line_cnt,"\r",
		
		title_text.write(x_text_tmp+'\n')
		
		x_num[line_cnt, :] = x_num_tmp
		x_lab[line_cnt, :] = x_lab_tmp
		x_text[line_cnt] = x_text_tmp
		y[line_cnt] = y_tmp
		line_cnt = line_cnt + 1


	return x_num, x_lab, x_text, y
