import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import random
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Template Localization Demo')
	parser.add_argument('proto', help='network prototxt')
	parser.add_argument('model', help='trained model')
	parser.add_argument('template_list', help='template image list')
	parser.add_argument('search_list', help='search image list')
	parser.add_argument('output_folder', help='output folder')
	parser.add_argument('--cpu', dest='cpu_mode', help='Use cpu mode', action='store_true')
	parser.set_defaults(cpu_mode=False)
	parser.add_argument('--test_num', type=int, default=50, help='test pair number')
	parser.add_argument('--overlap', type=float, default=0.3, help='overlap threshold for nms')
	parser.add_argument('--offset', type=int, default=0, help='offset of test id')
	parser.add_argument('--shuffle', help='shuffle test data', action='store_true')
	parser.set_defaults(shuffle=False)
	args = parser.parse_args()

	if not os.path.isfile(args.model):
		raise IOError(('Model not found: {:s}.\n').format(args.model))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()

	net = caffe.Net(args.proto, args.model, caffe.TEST)
	print 'Loaded network: ', args.model

	template_list, label_list1 = cu.readFileList(args.template_list, 1, 4)
	search_list, label_list2 = cu.readFileList(args.search_list, 1, 0)
	test_num = min(len(template_list), len(search_list), args.test_num)
	total_test_num = min(len(template_list), len(search_list))
	
	net.blobs['data_A'].reshape(1, 1, 64, 64)
	net.blobs['data_P'].reshape(1, 1, 256, 256)

	template_transformer = caffe.io.Transformer({'data': net.blobs['data_A'].data.shape})
	template_transformer.set_transpose('data', (2, 0, 1))
	search_transformer = caffe.io.Transformer({'data': net.blobs['data_P'].data.shape})
	search_transformer.set_transpose('data', (2, 0, 1))

	offset = args.offset
	if offset + test_num > total_test_num:
		offset = 0
	for i in range(test_num):
		id = i + offset
		if args.shuffle:
			id = random.randint(0, total_test_num - 1)
		template_path = template_list[id][0]
		search_path = search_list[id][0]
		template_filename = os.path.split(template_path)[1]
		search_filename = os.path.split(search_path)[1]
		template_img = cu.loadImageByCaffe(template_path, template_transformer, color=False)
		search_img = cu.loadImageByCaffe(search_path, search_transformer, color=False)

		net.blobs['data_A'].data[...] = template_img
		net.blobs['data_P'].data[...] = search_img
		output = net.forward()
		similarity_map = output['classifier/softmax_B'][0]

		template_origin_img = caffe.io.load_image(template_path)
		search_origin_img = caffe.io.load_image(search_path)
		save_path = template_filename + search_filename
		save_path = os.path.join(args.output_folder, save_path)
		cu.visLocalization(template_origin_img, search_origin_img, similarity_map, save_path, args.overlap)
		print "[", i, "] ", save_path