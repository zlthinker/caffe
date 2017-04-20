import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import time
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='match 3D features')
	parser.add_argument('proto', help='network prototxt')
	parser.add_argument('model', help='trained model')
	parser.add_argument('des_file1', help='descriptor file1')
	parser.add_argument('des_file2', help='descriptor file2')
	parser.add_argument('output_folder', help='output folder')
	parser.add_argument('--cpu', dest='cpu_mode', help='cpu mode', action='store_true')
	parser.set_defaults(cpu_mode=False)
	parser.add_argument('--batch_size', type=int, default=64, help='test batch size')
	parser.add_argument('--test_num', type=int, default=-1, help='test num')
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

	descriptors1 = np.load(args.des_file1)
	descriptors2 = np.load(args.des_file2)
	save_path = "match_pairs.txt"
	save_path = os.path.join(args.output_folder, save_path)
	f_out = open(save_path, 'w')
	timing_info = []

	time_start = time.time()
	true_num = 0
	test_num = min(args.test_num, descriptors1.shape[0])
	for i in range(0, test_num, 1):
		des1 = descriptors1[i, 1:]
		id1 = int(descriptors1[i][0])
		neg_pair = 0
		max_simi = 0
		max_simi_id = -1
		for b in range(0, batch_size):
			net.blobs['combine_A'].data[b, :, 0, 0] = des1
		for j in range(0, descriptors2.shape[0], batch_size):
			for b in range(j, min(j+batch_size, descriptors2.shape[0])):
				des2 = descriptors2[b, 1:]
				net.blobs['combine_P'].data[b-j, :, 0, 0] = des2

			output = net.forward()
			softmax= output['softmax']
			for b in range(j, min(j+batch_size, descriptors2.shape[0])):
				id2 = int(descriptors2[b][0])
				simi = softmax[b-j][1]
				if simi > max_simi:
					max_simi = simi
					max_simi_id = id2
		if max_simi > 0.95:
			print id1, max_simi_id, max_simi
			if id1 == max_simi_id:
				true_num = true_num + 1
			f_out.write(str(id1) + ' ' + str(max_simi_id) + ' ' + str(max_simi) + '\n')
	f_out.close()
	timing_info.append(('Match 3D feature descriptors', time.time() - time_start))
	cu.PrintRunningTime(timing_info)
	print true_num, 'matches are correct in', test_num, 'samples'



