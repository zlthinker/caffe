import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import time
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu

def readSiblingFileList(filepath):
    file_list = []
    file_list_p = []
    label_list = []
    with open(filepath) as file:
        for line in file:
            elements = line.strip().split(' ')
            file_list.append(elements[0])
            file_list_p.append(elements[1])
            label = float(line.strip().split(' ')[2])
            label_list.append(label)
    return file_list, file_list_p, label_list

def concatImages(imgs, label):
    img_num = len(imgs)
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]
    for img in imgs:
        if img.shape[0] > height:
            height = img.shape[0]
        if img.shape[1] > width:
            width = img.shape[1]
    concat = np.zeros((height, width * img_num, 3), dtype=np.float32)
    for i in range(img_num): 
        img = imgs[i]    
        img_left_top_y = (height - img.shape[0]) / 2
        img_left_top_x = (width - img.shape[1]) / 2 + width * i
        concat[img_left_top_y: img_left_top_y + img.shape[0], img_left_top_x: img_left_top_x + img.shape[1], :] = img
    if label > 0.5:
        cv2.rectangle(concat, (0, 0), (concat.shape[1], concat.shape[0]), (0, 1, 0), 4)
    else:
        cv2.rectangle(concat, (0, 0), (concat.shape[1], concat.shape[0]), (1, 0, 0), 4)
    return concat





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize image list')
    parser.add_argument('image_list', help='image list')
    args=parser.parse_args()

    image_list_path = args.image_list

    if not os.path.isfile(image_list_path):
        raise IOError('File not exists:' + image_list_path)

    image_list, label_list = cu.readFileList(image_list_path, 6, 1)
    image_num = len(image_list)

    fig = plt.figure()
    ax = fig.gca()
    fig.show()
    for i in range(image_num):
        imgs = []
        for path in image_list[i]:
            image = caffe.io.load_image(path)
            imgs.append(image)
        concat = concatImages(imgs, label_list[i][0])
        ax.imshow(concat)
        fig.canvas.draw()
        key = raw_input('press enter to continue, others to exit:\n')
        if key != '':
            exit(0)