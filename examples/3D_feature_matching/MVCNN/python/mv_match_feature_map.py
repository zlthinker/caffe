import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
import math
import random
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview Match Feature Map')
    parser.add_argument('proto', help='network prototxt')
    parser.add_argument('model', help='trained model')
    parser.add_argument('img_A1')
    parser.add_argument('img_A2')
    parser.add_argument('img_A3')
    parser.add_argument('img_P1')
    parser.add_argument('img_P2')
    parser.add_argument('img_P3')
    parser.add_argument('--cpu', dest='cpu_mode', help='Use cpu mode', action='store_true')
    parser.set_defaults(cpu_mode=False)
    parser.add_argument('--rescale', help='rescale activations within channel', action='store_true')
    parser.set_defaults(rescale=False)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise IOError(('Model not found: {:s}.\n').format(args.model))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    print 'Loaded network: ', args.model

    img_transformer = caffe.io.Transformer({'data': net.blobs['data_A1'].data.shape})
    img_transformer.set_transpose('data', (2, 0, 1))

    img_A1 = cu.loadImageByCaffe(args.img_A1, img_transformer, color=False)
    img_A2 = cu.loadImageByCaffe(args.img_A2, img_transformer, color=False)
    img_A3 = cu.loadImageByCaffe(args.img_A3, img_transformer, color=False)
    img_P1 = cu.loadImageByCaffe(args.img_P1, img_transformer, color=False)
    img_P2 = cu.loadImageByCaffe(args.img_P2, img_transformer, color=False)
    img_P3 = cu.loadImageByCaffe(args.img_P3, img_transformer, color=False)

    net.blobs['data_A1'].data[...] = img_A1
    net.blobs['data_A2'].data[...] = img_A2
    net.blobs['data_A3'].data[...] = img_A3
    net.blobs['data_P1'].data[...] = img_P1
    net.blobs['data_P2'].data[...] = img_P2
    net.blobs['data_P3'].data[...] = img_P3
    output = net.forward()
    feature_map_A = net.blobs['combine_A'].data[0, :, :, :]
    feature_map_P = net.blobs['combine_P'].data[0, :, :, :]

    # normalization
    # print feature_map_A[0].min(), feature_map_A[0].max()
    # print feature_map_A[1].min(), feature_map_A[1].max()
    # print feature_map_A[0]
    # print feature_map_A[1]
    # min = min(feature_map_A.min(), feature_map_P.min())
    # max = max(feature_map_A.max(), feature_map_P.max())
    # feature_map_A = (feature_map_A - feature_map_A.min()) / (feature_map_A.max() - feature_map_A.min())
    # feature_map_P = (feature_map_P - feature_map_P.min()) / (feature_map_P.max() - feature_map_P.min())
    # print feature_map_A[0]
    # print feature_map_P[0]
    if args.rescale:
        for c in range(feature_map_A.shape[0]):
            min_A = feature_map_A[c].min()
            max_A = feature_map_A[c].max()
            min_P = feature_map_P[c].min()
            max_P = feature_map_P[c].max()
            interval_A = max_A - min_A
            interval_P = max_P - min_P
            if interval_A == 0:
                interval_A = 1
            if interval_P == 0:
                interval_P = 1
            feature_map_A[c] = (feature_map_A[c] - min_A) / interval_A
            feature_map_P[c] = (feature_map_P[c] - min_P) / interval_P

    print feature_map_A.min(), feature_map_A.max()
    print feature_map_P.min(), feature_map_P.max()
    plt.figure(1)
    cu.visSquare(feature_map_A, feature_map_A.min(), feature_map_A.max())
    plt.figure(2)
    cu.visSquare(feature_map_P, feature_map_P.min(), feature_map_P.max())

    origin_A1 = caffe.io.load_image(args.img_A1)
    origin_A2 = caffe.io.load_image(args.img_A2)
    origin_A3 = caffe.io.load_image(args.img_A3)
    origin_P1 = caffe.io.load_image(args.img_P1)
    origin_P2 = caffe.io.load_image(args.img_P2)
    origin_P3 = caffe.io.load_image(args.img_P3)
    imgs_A = [origin_A1, origin_A2, origin_A3]
    imgs_P = [origin_P1, origin_P2, origin_P3]
    cu.visMVMatchPair(imgs_A, imgs_P, "", "", "", save=False)
    plt.show()