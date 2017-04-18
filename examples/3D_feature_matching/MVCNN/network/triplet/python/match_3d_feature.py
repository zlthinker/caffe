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
	args = parser.parse_args()

	descriptors1 = np.load(args.des_file1)
	descriptors2 = np.load(args.des_file2)
	save_path = "match_pairs.txt"
	save_path = os.path.join(args.output_folder, save_path)
	f_out = open(save_path, 'w')
	timing_info = []

	time_start = time.time()
	test_num = args.test_num
	if test_num == -1:
		test_num = descriptors1.shape[0]
	test_num = min(test_num, descriptors1.shape[0])
	print 'Feature number: ', descriptors1.shape[0], descriptors2.shape[0]
	for i in range(test_num):
		des1 = descriptors1[i, 1:]
		id1 = int(descriptors1[i][0])
		min_dist = sys.float_info.max
		min_id = -1
		sec_min_dist = sys.float_info.max
		sec_min_id = -1
		for j in range(descriptors2.shape[0]):
			des2 = descriptors2[j, 1:]
			id2 = int(descriptors2[j][0])
			dist = cu.L2distance(des1, des2, False)
			# print i, j, dist

			if dist < min_dist:
				sec_min_dist = min_dist
				sec_min_id = min_id
				min_dist = dist
				min_id = id2
			elif dist < sec_min_dist:
				sec_min_dist = dist
				sec_min_id = id2
			# 	print id1, id2, dist
			# if dist < 0.1:
			# 	print id1, id2, dist
			if id1 == id2:
				print id1, id2, dist, '********'
		# dist_ratio = min_dist / sec_min_dist
		# if dist_ratio > 0.7:
		# 	continue
		f_out.write(str(id1) + ' ' + str(min_id) + ' ' + str(min_dist) + '\n')
		if id1 == min_id:
			print 'Closet pair: ', id1, min_id, min_dist, sec_min_dist/min_dist, '******'
		else:
			print 'Closet pair: ', id1, min_id, min_dist, sec_min_dist/min_dist
		# print 'Second closet pair: ', id1, sec_min_id, sec_min_dist

	f_out.close()
	timing_info.append(('Match 3D feature descriptors', time.time() - time_start))
	cu.PrintRunningTime(timing_info)



