import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import scipy.stats as st
import scipy.ndimage.filters as fi
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu

def normalize(v, ord=2):
    norm=np.linalg.norm(v, ord)
    if norm==0: 
       return v
    return v/norm

def getGaussianKernel(kernlen, nsig):
	interval = (2*nsig+1.)/kernlen
	x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
	kern1d = np.diff(st.norm.cdf(x))
	kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
	kernel = kernel_raw/kernel_raw.sum()
	return kernel


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='encode 3D features')
	parser.add_argument('proto', help='network prototxt')
	parser.add_argument('model', help='trained model')
	parser.add_argument('file_list', help='testing file list')
	parser.add_argument('output_folder', help='output folder')
	parser.add_argument('--cpu', dest='cpu_mode', help='cpu mode', action='store_true')
	parser.set_defaults(cpu_mode=False)
	parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
	parser.add_argument('--test_num', type=int, default=-1, help='test num')
	parser.add_argument('--image_num', type=int, default=3, help='image num')
	parser.add_argument('--label_num', type=int, default=0, help='label num')
	parser.add_argument('--dim', type=int, default=512, help='feature dimension')
	parser.add_argument('--rescale', help='rescale activations within channel', action='store_true')
	parser.set_defaults(rescale=False)
	args = parser.parse_args()

	if not os.path.isfile(args.model):
		raise IOError(('Model not found: {:s}.\n').format(args.model))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()

	net = caffe.Net(args.proto, args.model, caffe.TEST)
	print 'Loaded network: ', args.model
	batch_size = args.batch_size
	if batch_size != net.blobs['data_A1'].data.shape[0]:
		raise Exception("Inconsistent batch size")

	file_list, label_list = cu.readFileList(args.file_list, args.image_num, args.label_num)
	test_num = args.test_num
	if test_num == -1:
		test_num = len(file_list)
	dim = args.dim

	img_transformer = caffe.io.Transformer({'data': net.blobs['data_A1'].data.shape})
	img_transformer.set_transpose('data', (2, 0, 1))

	timing_info = []

	time_start = time.time()
	descriptors = np.empty((test_num, dim+1), np.float32)
	rescale_method = cu.Rescale.MinMax
	if args.rescale:
		print 'Current rescale method is', rescale_method

	gkernel = getGaussianKernel(4, 2)
	print getGaussianKernel(4, 0.5)
	print 'Gaussian kernel', gkernel

	for i in range(test_num):
		files = file_list[i]
		img_A1 = cu.loadImageByCaffe(files[0], img_transformer, color=False)
		img_A2 = cu.loadImageByCaffe(files[1], img_transformer, color=False)
		img_A3 = cu.loadImageByCaffe(files[2], img_transformer, color=False)
		net.blobs['data_A1'].data[0, :, :, :] = img_A1
		net.blobs['data_A2'].data[0, :, :, :] = img_A2
		net.blobs['data_A3'].data[0, :, :, :] = img_A3
		output = net.forward()
		# out_des = net.blobs['combine_A'].data[0, :, :, :]
		feat = net.blobs['combine/conv1_A'].data[0, :, :, :]
		index = cu.parseTrackId(files[0])
		for c in range(feat.shape[0]):
			feat[c, :, :] = np.multiply(feat[c, :, :], gkernel)

		des = feat.flatten()
		des = normalize(des)
		des = np.insert(des, 0, np.float32(index))
		descriptors[i, :] = des

		if i % 1000 == 0:
			print i, 'images encoded.'
	timing_info.append(('Encode 3D feature descriptors #' + str(test_num), time.time() - time_start))

	save_start = time.time()
	save_path = "descriptors_"+str(dim)+".npy"
	save_path = os.path.join(args.output_folder, save_path)
	with open(save_path, 'w') as f_out:
		np.save(f_out, descriptors)
	duration = time.time() - save_start
	timing_info.append(('File save', duration))

	cu.PrintRunningTime(timing_info)
	print 'Average time is ', (time.time() - time_start) / float(test_num)

	# des_path = "descriptors_"+str(dim)+".txt"
	# des_path = os.path.join(args.output_folder, des_path)
	# with open(des_path, 'w') as f_des:
	# 	for i in range(descriptors.shape[0]):
	# 		des = descriptors[i]
	# 		f_des.write(str(int(des[0]))+' ')
	# 		for j in des[1:]:
	# 			f_des.write(str(j)+' ')
	# 		f_des.write('\n')
