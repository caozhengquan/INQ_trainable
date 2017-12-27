# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:13:51 2017

@author: fuwenyan
"""
import inspect
import os
import sys
sys.path.append("~/caffe-windows/python")
sys.path.append("~/caffe-windows/python/caffe")
import caffe
import struct
import numpy as np

def print_help():
    print """This script mainly serves as the basis of your customizations.
Customization is a must.
You can copy, paste, edit them in whatever way you want.
Usage:
    ./Decode_caffemodel.py deploy.prototxt model.caffemodel /where/to/save.txt """  
    sys.exit()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_help()
    else:
	# 使输出的参数完全显示
		# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
		np.set_printoptions(threshold='nan')

		# deploy文件
		MODEL_FILE =sys.argv[1]# 'deploy.prototxt'
		# 预先训练好的caffe模型
		PRETRAIN_FILE = sys.argv[2]#'model.caffemodel'

		# 保存参数的文件
		params_txt =sys.argv[3]# 'params.txt'
		pf_txt = open(params_txt, 'w')

		# 让caffe以测试模式读取网络参数
		net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
		print "net read OK"

		# 遍历每一层
		for param_name in net.params.keys():
		        print param_name                
			# 权重参数
			weight = net.params[param_name][0].data
		        print 'weight.shape',weight.shape
			# 偏置参数
                        layer_type=net.layer_dict[param_name].type
                        is_prelu = (layer_type=='PReLU')
                        is_conv = (layer_type == 'Convolution')
                        is_fc = (layer_type == 'InnerProduct')
                        print is_prelu
                        if is_prelu == 0:
			    bias = net.params[param_name][1].data
							 
		#    print 'weight.dtype',weight.dtype
		#    print 'bias.shape',bias.shape
			
			pf_dat = open('./weight/'+param_name+'.dat', 'wb')
			len_w=len(weight.shape)
			if (is_conv):##conv layer
				byte1=struct.pack('i',weight.shape[3])##由于write只能接受string或者buffer类型参数，因此对于其他类型要先通过struct进行转换
				byte3=struct.pack('i',weight.shape[1])
				byte4=struct.pack('i',weight.shape[0])
				pf_dat.write(byte1)
				pf_dat.write(byte3)
				pf_dat.write(byte4)
			elif(is_fc):##fc layer
				byte1=struct.pack('i',weight.shape[1])
				byte2=struct.pack('i',weight.shape[0])
				pf_dat.write(byte1)
				pf_dat.write(byte2)
                        elif(is_prelu):
                                byte=struct.pack('i',weight.shape[0])
                                pf_dat.write(byte)
			# 该层在prototxt文件中对应“top”的名称
			pf_txt.write(param_name)
			pf_txt.write('\n')

			# 写权重参数
			pf_txt.write('\n' + param_name + '_weight:\n\n')
			# 权重参数是多维数组，为了方便输出，转为单列数组
			
			weight.shape = (-1, 1)
		#    print 'weight.shape after:',weight.shape
				
			for w in weight:
				pf_txt.write('%f, ' % w)
				pf_dat.write(w)

                        if(is_prelu == 0):
			    # 写偏置参数
			    pf_txt.write('\n\n' + param_name + '_bias:\n\n')
			    # 偏置参数是多维数组，为了方便输出，转为单列数组
			    bias.shape = (-1, 1)
			    for b in bias:
				pf_txt.write('%f, ' % b)
				pf_dat.write(b)
			    pf_dat.close
		        pf_txt.write('\n\n')

		        pf_txt.close



