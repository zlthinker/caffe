import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import nms

def readFileList(filepath):
	file_list = []
	with open(filepath) as file:
		for line in file:
			path = line.strip().split(' ')[0]
			file_list.append(path)
	return file_list

def loadImage(image_path, transformer, color):
	img = caffe.io.load_image(image_path, color)
	img = (img * 255 - 128) * 0.00625
	transformed_image = transformer.preprocess('data', img)
	return transformed_image

def distance(vector1, vector2):
	if vector1.shape[0] != vector2.shape[0]:
		print "The dimension of two vectors are different."
		return 1
	norm1 = 0
	norm2 = 0
	for val in vector1:
		norm1 += val * val
	for val in vector2:
		norm2 += val * val
	norm1 = math.sqrt(norm1)
	norm2 = math.sqrt(norm2)
	dist = 0
	for i in range(vector1.shape[0]):
		dist += vector1[i] * vector2[i] / (norm1 * norm2)
	dist = 1 - dist
	return dist


def visualizeLocalization(img1, img2, simi_map, save_path, overlap):
	height = max(img1.shape[0], img2.shape[0])
	width = max(img1.shape[1], img2.shape[1])
	concat = np.zeros((height, width * 2, 3), dtype=np.float32)
	img1_left_top_y = (height - img1.shape[0]) / 2
	img1_left_top_x = (width - img1.shape[1]) / 2
	concat[img1_left_top_y : img1_left_top_y+img1.shape[0], img1_left_top_x : img1_left_top_x+img1.shape[1], :] = img1
	img2_left_top_y = (height - img2.shape[0]) / 2
	img2_left_top_x = (width - img2.shape[1]) / 2 + width
	concat[img2_left_top_y:img2_left_top_y+img2.shape[0], img2_left_top_x:img2_left_top_x+img2.shape[1], :] = img2
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.imshow(concat, aspect='equal')

	max_left_top_x = 0
	max_left_top_y = 0
	max_prob = 0

	boxes = np.empty(shape=[0, 5])
	for i in range(simi_map.shape[2]):
		for j in range(simi_map.shape[3]):
			prob = simi_map[0][0][i][j]
			if prob > 0.9:
				y = 8 * i
				x = 8 * j
				left_top_x = x + img2_left_top_x
				left_top_y = y + img2_left_top_y
				right_bottom_x = left_top_x + 64
				right_bottom_y = left_top_y + 64
				boxes = np.vstack([boxes, [left_top_x, left_top_y, right_bottom_x, right_bottom_y, prob]])

				if prob > max_prob:
					max_prob = prob
					max_left_top_x = left_top_x
					max_left_top_y = left_top_y

	pick_ids = nms.nms(boxes, overlap)
   
	for id in pick_ids:
		box = boxes[id, :]
		ax.add_patch(
					plt.Rectangle((box[0], box[1]),
					64, 64, fill=False, edgecolor='green', linewidth=3.5))
		ax.text(box[0], box[1] - 2,
						'{:.3f}'.format(box[4]),
						bbox=dict(facecolor='blue', alpha=0.5),
						fontsize=14, color='white')

	if max_prob > 0:
		ax.add_patch(
					plt.Rectangle((max_left_top_x, max_left_top_y),
					64, 64, fill=False, edgecolor='red', linewidth=3.5))
		ax.text(max_left_top_x, max_left_top_y - 2,
						'{:.3f}'.format(max_prob),
						bbox=dict(facecolor='blue', alpha=0.5),
						fontsize=14, color='red')
	# ax.set_title(prob, fontsize=64)

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	plt.savefig(save_path)


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
	parser.add_argument('--shuffle', help='Use cpu mode', action='store_true')
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

	template_list = readFileList(args.template_list)
	search_list = readFileList(args.search_list)
	if not len(template_list) == len(search_list):
		 logging.error('template list and search list have diff size.\n')
	test_num = min(len(template_list), args.test_num)
	total_test_num = len(template_list)

	net.blobs['data_template'].reshape(1, 1, 64, 64)
	net.blobs['data_search'].reshape(1, 1, 256, 256)

	template_transformer = caffe.io.Transformer({'data': net.blobs['data_template'].data.shape})
	template_transformer.set_transpose('data', (2, 0, 1))
	search_transformer = caffe.io.Transformer({'data': net.blobs['data_search'].data.shape})
	search_transformer.set_transpose('data', (2, 0, 1))

	for i in range(test_num):
		id = i
		if args.shuffle:
			id = random.randint(0, total_test_num - 1)
		template_path = template_list[id]
		search_path = search_list[id]
		template_filename = os.path.split(template_path)[1]
		search_filename = os.path.split(search_path)[1]
		template_img = loadImage(template_path, template_transformer, color=False)
		search_img = loadImage(search_path, search_transformer, color=False)

		net.blobs['data_template'].data[...] = template_img
		net.blobs['data_search'].data[...] = search_img
		output = net.forward()
		similarity_map = output['classifier/softmax_B']

		template_origin_img = caffe.io.load_image(template_path)
		search_origin_img = caffe.io.load_image(search_path)
		save_path = template_filename + search_filename
		save_path = os.path.join(args.output_folder, save_path)
		visualizeLocalization(template_origin_img, search_origin_img, similarity_map, save_path, args.overlap)