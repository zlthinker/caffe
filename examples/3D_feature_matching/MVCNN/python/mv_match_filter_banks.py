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

    # 0 for weights, 1 for biases; select only 0, 1, 2 these three input channels
    filters = net.params['A1/conv4'][0].data[:, :, :, :]  
    filters = filters[:, 0:min(3, filters.shape[1]), :, :]
    cu.visSquare(filters.transpose(0, 2, 3, 1))
    plt.show()