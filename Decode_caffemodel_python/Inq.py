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
import math

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
                        pf_inq = open('./weight/'+param_name+'.inq','wb')
			len_w=len(weight.shape)
			if (is_conv):##conv layer
				byte1=struct.pack('i',weight.shape[3])##由于write只能接受string或者buffer类型参数，因此对于其他类型要先通过struct进行转换
				byte3=struct.pack('i',weight.shape[1])
				byte4=struct.pack('i',weight.shape[0])
				pf_dat.write(byte1)
				pf_dat.write(byte3)
				pf_dat.write(byte4)
				pf_inq.write(byte1)
				pf_inq.write(byte3)
				pf_inq.write(byte4)
			elif(is_fc):##fc layer
				byte1=struct.pack('i',weight.shape[1])
				byte2=struct.pack('i',weight.shape[0])
				pf_dat.write(byte1)
				pf_dat.write(byte2)
				pf_inq.write(byte1)
				pf_inq.write(byte2)
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
			w_w_buf=[1]*100000
                        w_s_buf=[1]*100000
                        idx=1
			for w in weight:
                                w_sign=1
                                if w<0:
                                    w_sign=0
                                w_v=0
                                if w !=0:
                                    w_v=math.log(abs(w),2)
                                w_w=w_v#+7
                                w_w_buf[idx]=w_w
                                w_s_buf[idx]=w_sign

				pf_txt.write('%d, ' % w_w)
                                idx=idx+1
				pf_dat.write(w)
                        pf_txt.write('\n\n\n\n')
                        #print(max(w_w_buf))
                        print(min(w_w_buf))
                        min_w=min(w_w_buf)
                        byte_min=struct.pack('i',min_w)
                        pf_inq.write(byte_min)
                        for w_i in range(1,idx):
                            pf_txt.write('%d, '% (w_w_buf[w_i]-min_w))
                        pf_txt.write('\n\n\n\n\n\n')
                        for s_real in range(1,(idx)/8):
                            bit_8 = w_s_buf[8*s_real-7]*128
                            bit_7 = w_s_buf[8*s_real-6]*64
                            bit_6 = w_s_buf[8*s_real-5]*32
                            bit_5 = w_s_buf[8*s_real-4]*16
                            bit_4 = w_s_buf[8*s_real-3]*8
                            bit_3 = w_s_buf[8*s_real-2]*4
                            bit_2 = w_s_buf[8*s_real-1]*2
                            bit_1 = w_s_buf[8*s_real]
                            s_real_v=bit_8+bit_7+bit_6+bit_5+bit_4+bit_3+bit_2+bit_1
                            pf_txt.write('%d, '%s_real_v)
                            byte_s_real=struct.pack('B',(s_real_v))
                            pf_inq.write(byte_s_real)
                        bit_left=0
                        for s_real in range(idx-idx%8,idx):
                            print(bit_left)
                            print(w_s_buf[s_real])
                            print(7-s_real+idx-idx%8)
                            bit_left=bit_left+w_s_buf[s_real]*2**(7-s_real+idx/8)
                        print(bit_left)
                        byte_s_real=struct.pack('B',bit_left)
                        pf_inq.write(byte_s_real)


                        pf_txt.write('\n\n\n\n')
                        for w_real in range(1,(idx)/2):
                            high_v = (w_w_buf[2*w_real-1]-min_w)*16
                            low_v = w_w_buf[2*w_real]-min_w
                            pf_txt.write('%d, '%(high_v+low_v))
                            byte_w_real=struct.pack('B',(high_v+low_v))
                            pf_inq.write(byte_w_real)

                        if idx%2 ==0:
                            pf_txt.write('%d'%(w_w_buf[idx]-min_w))
                            byte_w_real = struct.pack('B',w_w_buf[idx]-min_w)
                            pf_inq.write(byte_w_real)
                        


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



