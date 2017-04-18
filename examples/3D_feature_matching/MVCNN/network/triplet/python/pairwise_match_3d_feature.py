import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='multi-view match test')
	parser.add_argument('proto', help='network prototxt')
	parser.add_argument('model', help='trained model')
	parser.add_argument('file_list', help='testing file list')
	parser.add_argument('output_folder', help='output folder')
	parser.add_argument('--cpu', dest='cpu_mode', help='cpu mode', action='store_true')
	parser.set_defaults(cpu_mode=False)
	parser.add_argument('--test_num', type=int, default=200, help='test num')
	args = parser.parse_args()

	if not os.path.isfile(args.model):
		raise IOError(('Model not found: {:s}.\n').format(args.model))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()

	net = caffe.Net(args.proto, args.model, caffe.TEST)
	print 'Loaded network: ', args.model

	file_list, label_list = cu.readFileList(args.file_list, 9, 0)
	test_num = min(len(file_list), args.test_num)

	img_transformer = caffe.io.Transformer({'data': net.blobs['data_A1'].data.shape})
	img_transformer.set_transpose('data', (2, 0, 1))

	for i in range(test_num):
		files = file_list[i]
		# gt_label = label_list[i][0]
		img_A1 = cu.loadImageByCaffe(files[0], img_transformer, color=False)
		img_A2 = cu.loadImageByCaffe(files[1], img_transformer, color=False)
		img_A3 = cu.loadImageByCaffe(files[2], img_transformer, color=False)
		img_P1 = cu.loadImageByCaffe(files[3], img_transformer, color=False)
		img_P2 = cu.loadImageByCaffe(files[4], img_transformer, color=False)
		img_P3 = cu.loadImageByCaffe(files[5], img_transformer, color=False)

		net.blobs['data_A1'].data[...] = img_A1
		net.blobs['data_A2'].data[...] = img_A2
		net.blobs['data_A3'].data[...] = img_A3
		net.blobs['data_P1'].data[...] = img_P1
		net.blobs['data_P2'].data[...] = img_P2
		net.blobs['data_P3'].data[...] = img_P3
		output = net.forward()
		feat_A = net.blobs['feat_A'].data[0]
		feat_P = net.blobs['feat_P'].data[0]
		dist = cu.L2distance(feat_A, feat_P)


		origin_A1 = caffe.io.load_image(files[0])
		origin_A2 = caffe.io.load_image(files[1])
		origin_A3 = caffe.io.load_image(files[2])
		origin_P1 = caffe.io.load_image(files[3])
		origin_P2 = caffe.io.load_image(files[4])
		origin_P3 = caffe.io.load_image(files[5])
		imgs_A = [origin_A1, origin_A2, origin_A3]
		imgs_P = [origin_P1, origin_P2, origin_P3]
		save_path = str(i)+'.jpg'
		save_path = os.path.join(args.output_folder, save_path)
		cu.visMVMatchPair(imgs_A, imgs_P, dist, "", save_path)
		print '[', i, ']', save_path
