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
	parser.add_argument('des_file1', help='descriptor file1')
	parser.add_argument('des_file2', help='descriptor file2')
	parser.add_argument('output_folder', help='output folder')
	parser.add_argument('--test_num', type=int, default=-1, help='test num')
	parser.add_argument('--start', type=int, default=0, help='start of feature idx')
	parser.add_argument('--min_ratio', type=float, default=1.5, help='threshold for ratio test')
	args = parser.parse_args()

	min_ratio = args.min_ratio

	repre1 = np.load(args.des_file1)
	repre2 = np.load(args.des_file2)
	descriptors1 = np.delete(repre1, 0, 1)	# erase id
	descriptors2 = np.delete(repre2, 0, 1)
	save_path = "match_pairs.txt"
	save_path = os.path.join(args.output_folder, save_path)
	f_out = open(save_path, 'w')

	pos_save_path = "pos_match_pairs.txt"
	pos_save_path = os.path.join(args.output_folder, pos_save_path)
	f_pos = open(pos_save_path, 'w')

	neg_save_path = "neg_match_pairs.txt"
	neg_save_path = os.path.join(args.output_folder, neg_save_path)
	f_neg = open(neg_save_path, 'w')

	timing_info = []

	time_start = time.time()
	test_num = args.test_num
	if test_num == -1:
		test_num = descriptors1.shape[0]
	test_num = min(test_num, descriptors1.shape[0])
	print 'Feature number: ', descriptors1.shape[0], descriptors2.shape[0]
	start = min(args.start, test_num)

	num_true_pos = 0
	for i in range(start, test_num, 1):
		des1 = descriptors1[i]
		id1 = int(repre1[i][0])
		des1_tile = np.tile(des1, (descriptors2.shape[0], 1))
		des_sub = des1_tile - descriptors2
		des_sub_square = np.square(des_sub)
		distances = np.sum(des_sub_square, axis=1)

		min_dist = sys.float_info.max
		min_id = -1
		sec_min_dist = sys.float_info.max
		sec_min_id = -1
		for j in range(descriptors2.shape[0]):
			dist = distances[j]
			id2 = int(repre2[j][0])
			# print i, j, dist

			if dist < min_dist:
				sec_min_dist = min_dist
				sec_min_id = min_id
				min_dist = dist
				min_id = id2
			elif dist < sec_min_dist:
				sec_min_dist = dist
				sec_min_id = id2

		if id1 == min_id:
			num_true_pos += 1
		dist_ratio = sec_min_dist / min_dist
		if dist_ratio > min_ratio:
			f_out.write(str(id1) + ' ' + str(min_id) + ' ' + str(min_dist) + '\n')
			print '[', i+1, num_true_pos, ']', id1, min_id, min_dist, dist_ratio, '*************'
		else:
			print '[', i+1, num_true_pos, ']', id1, min_id, min_dist, dist_ratio
		if id1 == min_id:
		# 	print 'Closet pair: ', id1, min_id, min_dist, dist_ratio, '******'
			f_pos.write(str(id1) + ' ' + str(min_id) + ' ' + str(min_dist) + ' ' + str(dist_ratio) + '\n')
		else:
		# 	print 'Closet pair: ', id1, min_id, min_dist, dist_ratio
			f_neg.write(str(id1) + ' ' + str(min_id) + ' ' + str(min_dist) + ' ' + str(dist_ratio) + '\n')

	f_out.close()
	f_pos.close()
	f_neg.close()
	timing_info.append(('Match 3D feature descriptors', time.time() - time_start))
	cu.PrintRunningTime(timing_info)



