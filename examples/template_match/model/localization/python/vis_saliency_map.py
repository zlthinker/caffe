import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import random
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu

def vis_square(data, min, max):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - min)/(max - min)
    
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

    net = caffe.Net(args.proto, args.model, caffe.TRAIN)
    print 'Loaded network: ', args.model
    
    net.blobs['data_A'].reshape(1, 1, 64, 64)
    net.blobs['data_P'].reshape(1, 1, 256, 256)

    template_transformer = caffe.io.Transformer({'data': net.blobs['data_A'].data.shape})
    template_transformer.set_transpose('data', (2, 0, 1))
    search_transformer = caffe.io.Transformer({'data': net.blobs['data_P'].data.shape})
    search_transformer.set_transpose('data', (2, 0, 1))

    template_path = args.template_image
    search_path = args.search_image
    template_img = cu.loadImageByCaffe(template_path, template_transformer, color=False)
    search_img = cu.loadImageByCaffe(search_path, search_transformer, color=False)

    net.blobs['data_A'].data[...] = template_img
    net.blobs['data_P'].data[...] = search_img
    classifier_label = np.ones((1, 1, 1, 1))
    net.blobs['label'] = classifier_label
    output = net.forward()

    diff = np.ones((1, 2, 1, 1))

    diffs = net.backward(start='classifier/conv2', end='conv0', **{'classifier/conv2': diff})
    print diffs.keys()
    exit(0)
    diff_A = diffs['data_A'][0]
    print diff_A.shape
    print diff_A.min()
    print diff_A.max()
    exit(0)
    cu.visSquare(diff_A, diff_A.min(), diff_A.max())
    plt.show()
    for key, val in bw.iteritems():
        print key
    exit(0)

    template_origin_img = caffe.io.load_image(template_path)
    search_origin_img = caffe.io.load_image(search_path)
    plt.figure(1)
    plt.imshow(template_origin_img)
    plt.figure(2)
    plt.imshow(search_origin_img)
    featuremap_template = net.blobs['pool4'].data[0, :64]
    featuremap_search = net.blobs['pool4_p'].data[0, :64]
    min = min(featuremap_template.min(), featuremap_search.min())
    max = max(featuremap_template.max(), featuremap_search.max())
    plt.figure(3)
    cu.visSquare(featuremap_template, min, max)
    plt.figure(4)
    cu.visSquare(featuremap_search, min, max)
    plt.show()