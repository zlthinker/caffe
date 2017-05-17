import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy.spatial.distance import cdist
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
	parser.add_argument('--save', default=False, help='threshold for ratio test', action='store_true')
	args = parser.parse_args()

	min_ratio = args.min_ratio
	repre1 = np.load(args.des_file1)
	repre2 = np.load(args.des_file2)
	descriptors1 = np.delete(repre1, 0, 1)	# erase id
	descriptors2 = np.delete(repre2, 0, 1)

	save_path = "match_pairs.txt"
	save_path = os.path.join(args.output_folder, save_path)
	f_out = open(save_path, 'w')

	dist_save_path = 'distances.npy'
	dist_save_path = os.path.join(args.output_folder, dist_save_path)

	timing_info = []
	time_start = time.time()

	if os.path.exists(dist_save_path):
		print 'Load distance file:', dist_save_path
		exh_distances = np.load(dist_save_path)
		argmins1 = np.argmin(exh_distances, axis=1)	#min id of each row
		argmins2 = np.argmin(exh_distances, axis=0) #min id of each col

		for i in range(descriptors1.shape[0]):
			argmin1 = argmins1[i]
			argmin2 = argmins2[argmin1]
			min_dist = exh_distances[i].min()
			new_distances = np.delete(exh_distances[i], [argmin1])
			sec_min_dist = new_distances.min()
			ratio = sec_min_dist / min_dist
			if i != argmin2 or ratio < min_ratio:
				continue
			id1 = int(repre1[i][0])
			id2 = int(repre2[argmin1][0])
			f_out.write(str(id1) + ' ' + str(id2) + '\n')
			# print id1, id2, ratio

	else:
		pos_save_path = "pos_match_pairs.txt"
		pos_save_path = os.path.join(args.output_folder, pos_save_path)
		f_pos = open(pos_save_path, 'w')
		neg_save_path = "neg_match_pairs.txt"
		neg_save_path = os.path.join(args.output_folder, neg_save_path)
		f_neg = open(neg_save_path, 'w')

		test_num = args.test_num
		if test_num == -1:
			test_num = descriptors1.shape[0]
		test_num = min(test_num, descriptors1.shape[0])
		print 'Feature number: ', descriptors1.shape[0], descriptors2.shape[0]
		start = min(args.start, test_num)

		metric = "euclidean"
		print "Computing exhaustive", metric, "distances... "
		exh_distances = cdist(descriptors1, descriptors2, metric=metric)


		num_true_pos = 0
		for i in range(start, test_num, 1):
			des1 = descriptors1[i]
			id1 = int(repre1[i][0])
			distances = exh_distances[i, :]

			min_dist = distances.min()
			argmin = np.argmin(distances)
			new_distances = np.delete(distances, [argmin])
			sec_min_dist = new_distances.min()

			min_id = int(repre2[argmin][0])

			if id1 == min_id:
				num_true_pos += 1
			dist_ratio = sec_min_dist / min_dist

			print '[', i+1, num_true_pos, ']', id1, min_id, min_dist, dist_ratio
			if id1 == min_id:
			# 	print 'Closet pair: ', id1, min_id, min_dist, dist_ratio, '******'
				f_pos.write(str(id1) + ' ' + str(min_id) + ' ' + str(min_dist) + ' ' + str(dist_ratio) + '\n')
			else:
			# 	print 'Closet pair: ', id1, min_id, min_dist, dist_ratio
				f_neg.write(str(id1) + ' ' + str(min_id) + ' ' + str(min_dist) + ' ' + str(dist_ratio) + '\n')
		f_pos.close()
		f_neg.close()
		if args.save:
			with open(dist_save_path, 'w') as f_dist:
				np.save(f_dist, exh_distances)

	timing_info.append(('Match 3D feature descriptors', time.time() - time_start))
	cu.PrintRunningTime(timing_info)



