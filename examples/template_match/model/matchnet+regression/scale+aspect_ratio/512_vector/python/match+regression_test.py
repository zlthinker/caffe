import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math

def readFileList(filepath):
    file_list = []
    labels = []
    with open(filepath) as file:
        for line in file:
            elements = line.strip().split(' ')
            path = elements[0]
            file_list.append(path)
            if len(elements) == 3:
                labels.append([elements[1], elements[2]])
    return file_list, labels

def loadImage(image_path, transformer, color):
    img = caffe.io.load_image(image_path, color)
    img = (img * 255 - 128) * 0.00625
    transformed_image = transformer.preprocess('data', img)
    return transformed_image

def visualizePair(img1, img2, prob, regression, gt_boxes, save_path):
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])
    concat = np.zeros((height, width * 2, 3), dtype=np.float32)
    img1_left_top_y = (height - img1.shape[0]) / 2
    img1_left_top_x = (width - img1.shape[1]) / 2
    concat[img1_left_top_y : img1_left_top_y+img1.shape[0], img1_left_top_x : img1_left_top_x+img1.shape[1], :] = img1
    img2_left_top_y = (height - img2.shape[0]) / 2
    img2_left_top_x = (width - img2.shape[1]) / 2 + width
    concat[img2_left_top_y:img2_left_top_y+img2.shape[0], img2_left_top_x:img2_left_top_x+img2.shape[1], :] = img2
    fig, ax = plt.subplots(figsize=(12, 12)) #create figure
    ax.set_title(prob, fontsize=64)
    ax.imshow(concat, aspect='equal')

    # draw bounding box
    center_x = width * 3 / 2
    center_y = height / 2
    width_scale = regression[0]
    height_scale = width_scale
    if len(regression) == 2:
        height_scale = regression[1]
    box_width = width / width_scale
    box_height = height / height_scale
    ax.add_patch(plt.Rectangle((center_x - box_width / 2, center_y - box_height / 2),
                    box_width, box_height, fill=False, edgecolor='green', linewidth=3.5))

    gt_width_scale = float(gt_boxes[0])
    gt_height_scale = float(gt_boxes[1])
    gt_box_width = width / gt_width_scale
    gt_box_height = height / gt_height_scale
    ax.add_patch(plt.Rectangle((center_x - gt_box_width / 2, center_y - gt_box_height / 2),
                    gt_box_width, gt_box_height, fill=False, edgecolor='red', linewidth=3.5))

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    # print 'save image: ', save_path
    print width_scale, ' ', height_scale, ' ', gt_width_scale, ' ', gt_height_scale




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch Match Demo')
    parser.add_argument('proto', help='network prototxt')
    parser.add_argument('model', help='trained model')
    parser.add_argument('template_list', help='template image list')
    parser.add_argument('search_list', help='search image list')
    parser.add_argument('output_folder', help='output folder')
    parser.add_argument('--cpu', dest='cpu_mode', help='Use cpu mode', action='store_true')
    parser.set_defaults(cpu_mode=False)
    parser.add_argument('--test_num', type=int, default=200, help='test pair number')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise IOError(('Model not found: {:s}.\n').format(args.model))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    print 'Loaded network: ', args.model

    template_list, gt_boxes = readFileList(args.template_list)
    search_list, no_use = readFileList(args.search_list)
    if not len(template_list) == len(search_list):
         logging.error('template list and search list have diff size.\n')
    test_num = min(len(template_list), args.test_num)

    net.blobs['data_A'].reshape(1, 1, 64, 64)
    net.blobs['data_P'].reshape(1, 1, 64, 64)

    template_transformer = caffe.io.Transformer({'data': net.blobs['data_A'].data.shape})
    template_transformer.set_transpose('data', (2, 0, 1))
    search_transformer = caffe.io.Transformer({'data': net.blobs['data_P'].data.shape})
    search_transformer.set_transpose('data', (2, 0, 1))

    true_num = 0
    for i in range(test_num):
        template_path = template_list[i]
        search_path = search_list[i]
        template_filename = os.path.split(template_path)[1]
        search_filename = os.path.split(search_path)[1]
        template_img = loadImage(template_path, template_transformer, color=False)
        search_img = loadImage(search_path, search_transformer, color=False)

        net.blobs['data_A'].data[...] = template_img
        net.blobs['data_P'].data[...] = search_img
        output = net.forward()
        softmax_prob = output['softmax'][0]
        label = False
        if softmax_prob[0] < softmax_prob[1]:
            label = True
            true_num += 1
        print '[', i, ']', template_filename, ' match ', search_filename, ': ', label
        prob = "{0:.2f}".format(float(softmax_prob[1]))
        regression = output['regression/conv2'][0]
        template_origin_img = caffe.io.load_image(template_path)
        search_origin_img = caffe.io.load_image(search_path)
        save_path = template_filename + search_filename
        save_path = os.path.join(args.output_folder, save_path)
        visualizePair(template_origin_img, search_origin_img, prob, regression, gt_boxes[i], save_path)

    print "The percentage of true num is ", float(true_num) / test_num




