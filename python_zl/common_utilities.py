import os, sys, cv2
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import nms

#how to use it:
#sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
#import common_utilities as cu

def L2distance(vector1, vector2):
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

def readFileList(filepath, file_num, label_num):
    file_list_list = []
    label_list_list = []
    with open(filepath) as file:
        for line in file:
            elements = line.strip().split(' ')
            if file_num + label_num != len(elements):
            	raise Exception("Inconsistent number of elements!")
            file_list = []
            label_list = []
            for i in range(file_num):
            	file_list.append(elements[i])
            for i in range(label_num):
            	label = float(elements[i + file_num])
            	label_list.append(label)
            file_list_list.append(file_list)
            label_list_list.append(label_list)
    return file_list_list, label_list_list


def loadImageByCaffe(image_path, transformer, color):
    img = caffe.io.load_image(image_path, color)
    img = (img * 255 - 128) * 0.00625
    transformed_image = transformer.preprocess('data', img)
    return transformed_image

def visSquare(data, min, max):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - min) / (max - min)
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

def visMatchNetPair(img1, img2, label, save_path):
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])
    concat = np.zeros((height, width * 2, 3), dtype=np.float32)
    img1_left_top_y = (height - img1.shape[0]) / 2
    img1_left_top_x = (width - img1.shape[1]) / 2
    concat[img1_left_top_y : img1_left_top_y+img1.shape[0], img1_left_top_x : img1_left_top_x+img1.shape[1], :] = img1
    img2_left_top_y = (height - img2.shape[0]) / 2
    img2_left_top_x = (width - img2.shape[1]) / 2 + width
    concat[img2_left_top_y:img2_left_top_y+img2.shape[0], img2_left_top_x:img2_left_top_x+img2.shape[1], :] = img2
    fig, ax = plt.subplots(figsize=(6, 3.5)) #create figure 600x350
    ax.set_title(label, fontsize=32)
    ax.imshow(concat)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    print 'save image: ', save_path

def visLocalization(img1, img2, simi_map, save_path, overlap):
	height = max(img1.shape[0], img2.shape[0])
	width = max(img1.shape[1], img2.shape[1])
	concat = np.zeros((height, width * 2, 3), dtype=np.float32)
	img1_left_top_y = (height - img1.shape[0]) / 2
	img1_left_top_x = (width - img1.shape[1]) / 2
	concat[img1_left_top_y : img1_left_top_y+img1.shape[0], img1_left_top_x : img1_left_top_x+img1.shape[1], :] = img1
	img2_left_top_y = (height - img2.shape[0]) / 2
	img2_left_top_x = (width - img2.shape[1]) / 2 + width
	concat[img2_left_top_y:img2_left_top_y+img2.shape[0], img2_left_top_x:img2_left_top_x+img2.shape[1], :] = img2
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.imshow(concat, aspect='equal')

	max_left_top_x = 0
	max_left_top_y = 0
	max_prob = 0

	boxes = np.empty(shape=[0, 5])
	for i in range(simi_map.shape[1]):
		for j in range(simi_map.shape[2]):
			prob = simi_map[0][i][j]
			if prob > 0.8:
				y = 8 * i
				x = 8 * j
				width_patch = 64
				height_patch = 64
				# print width_scale, ' ', height_scale
				left_top_x = x + img2_left_top_x
				left_top_y = y + img2_left_top_y
				right_bottom_x = left_top_x + width_patch
				right_bottom_y = left_top_y + height_patch
				# print left_top_x, ' ', left_top_y
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
					box[2] - box[0], box[3] - box[1], fill=False, edgecolor='green', linewidth=2))
		ax.text(box[0], box[1] - 2,
						'{:.3f}'.format(box[4]),
						bbox=dict(facecolor='blue', alpha=0.5),
						fontsize=12, color='white')

	if max_prob > 0:
		ax.add_patch(
					plt.Rectangle((max_left_top_x, max_left_top_y),
					box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
		ax.text(max_left_top_x, max_left_top_y - 2,
						'{:.3f}'.format(max_prob),
						bbox=dict(facecolor='blue', alpha=0.5),
						fontsize=12, color='red')
	ax.set_title(max_prob, fontsize=32)

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	plt.savefig(save_path)