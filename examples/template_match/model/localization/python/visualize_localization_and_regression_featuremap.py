import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import nms
import visualize_localization_and_regression as vis

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Template Localization Demo')
    parser.add_argument('proto', help='network prototxt')
    parser.add_argument('model', help='trained model')
    parser.add_argument('template_image', help='template image path')
    parser.add_argument('search_image', help='search image path')
    parser.add_argument('--cpu', dest='cpu_mode', help='Use cpu mode', action='store_true')
    parser.set_defaults(cpu_mode=False)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise IOError(('Model not found: {:s}.\n').format(args.model))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    print 'Loaded network: ', args.model
    
    net.blobs['data_template'].reshape(1, 1, 64, 64)
    net.blobs['data_search'].reshape(1, 1, 256, 256)

    template_transformer = caffe.io.Transformer({'data': net.blobs['data_template'].data.shape})
    template_transformer.set_transpose('data', (2, 0, 1))
    search_transformer = caffe.io.Transformer({'data': net.blobs['data_search'].data.shape})
    search_transformer.set_transpose('data', (2, 0, 1))

    template_path = args.template_image
    search_path = args.search_image
    template_img = vis.loadImage(template_path, template_transformer, color=False)
    search_img = vis.loadImage(search_path, search_transformer, color=False)

    net.blobs['data_template'].data[...] = template_img
    net.blobs['data_search'].data[...] = search_img
    output = net.forward()

    template_origin_img = caffe.io.load_image(template_path)
    search_origin_img = caffe.io.load_image(search_path)
    plt.figure(1)
    plt.imshow(template_origin_img)
    plt.figure(2)
    plt.imshow(search_origin_img)
    featuremap_template = net.blobs['pool4'].data[0, :9]
    featuremap_search = net.blobs['st/pool4'].data[0, :9]
    plt.figure(3)
    vis_square(featuremap_template)
    plt.figure(4)
    vis_square(featuremap_search)
    plt.show()