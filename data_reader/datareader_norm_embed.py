#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import re

# read raw data
def read_raw_data( file_path, data_length ):
	x_num_index = [15, 21, 23, 29, 31, 33, 35, 37, 39, 43, 49, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 77, 79, 85, 87, 89, 91, 93]
	# 602 [channelFstLevel] : [0,45]	len[46]
	# 603 [titleImageSetNum] : [0,142]	len[143]
	# 606 [channelFstTopicMax] : [0,129]	len[130]
	# 607 [titleWordNum] : [0,32]		len[33]
	# 617 [userSex] : [0,2]			len[3]
	# 619 [showListLoc]: [1,20]		len[20]
	# 645 [titleByteNum]: [0,117]		len[118]
	x_lab_index = [17, 19, 25, 27, 47, 51, 81]
	# 502 [titleText] : [string length=100]
	x_text_index= [5]
	y_index = [1]

	x_num = np.zeros(shape=(data_length, len(x_num_index)))
	x_lab = np.zeros(shape=(data_length, 493))
	x_text = np.full(shape=(data_length, 1), fill_value='', dtype='|S100')
	y = np.zeros(shape=(data_length, 1))	

	data_file = open(file_path, 'r')
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
		
		x_num[line_cnt, :] = x_num_tmp
		x_lab[line_cnt, 0:46] = np.eye(46)[x_lab_tmp[0]]
		x_lab[line_cnt, 46:189] = np.eye(143)[x_lab_tmp[1]]
		x_lab[line_cnt, 189:319] = np.eye(130)[x_lab_tmp[2]]
		x_lab[line_cnt, 319:352] = np.eye(33)[x_lab_tmp[3]]
		x_lab[line_cnt, 352:355] = np.eye(3)[x_lab_tmp[4]]
		x_lab[line_cnt, 355:375] = np.eye(20)[x_lab_tmp[5]-1]
		x_lab[line_cnt, 375:493] = np.eye(118)[x_lab_tmp[6]]
		x_text[line_cnt] = x_text_tmp
		y[line_cnt] = y_tmp
		line_cnt = line_cnt + 1
		print line_cnt,'\r',

		mean_number = [27670050099646.695, 737.50909937500001, 0.73641900000000005, 1.09050325, 3.2547649999999999, 0.18988734424661663, 0.018218518688147518, 0.58289793214690711, 58525.112806500001, 0.0032351559517227253, 2.1508435000000001, 0.00047137500000000001, 0.10948682168147734, 0.10056244604570011, 0.10037418390206244, 0.1128079657436573, 53497.460104749996, 6633027.3096455, 2806267.0190227502, 140148.42415524999, 0.0047422759402424982, 0.0039007925999112515, 0.0060802298141032487, 0.0047484130137352473, 0.99914212499999999, 13095.282943, -0.0089626250000000001, 0.14179575, 0.00088750000000000005]
		std_number = [22592484274716896.0, 909.78373565381014, 0.4405746888315305, 0.56558501725155141, 0.95829499360843995, 0.16766438883598445, 0.019217202331222292, 0.25208077216449359, 160188.84980930071, 0.0087498100797740019, 19.489922523922644, 0.021706054584133341, 0.076003425744698247, 0.024476119567452954, 0.024967745644477117, 0.049456851743592344, 134435.62475140626, 4479273.7576365927, 2594456.424606265, 249033.09038354468, 0.013363847394051642, 0.011820077852673413, 0.012348090381514064, 0.011191130657562625, 0.029276937177313746, 101498.45848874416, 1.3804452348257468, 1.7717601319823015, 0.02977771555626791]
	for i in range(len(x_num_index)):
		x_num[:, i] = (x_num[:, i] - mean_number[i]) / std_number[i]

	return x_num, x_lab, x_text, y
