import os, sys, cv2, logging
import argparse
import caffe
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe/python_zl/')
import common_utilities as cu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch Match Demo')
    parser.add_argument('proto', help='network prototxt')
    parser.add_argument('model', help='trained model')
    parser.add_argument('file_list', help='sibling image list')
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

    file_list, label_list = cu.readFileList(args.file_list, 2, 1)
    test_num = min(len(file_list), args.test_num)

    net.blobs['data_A'].reshape(1, 1, 64, 64)
    net.blobs['data_P'].reshape(1, 1, 64, 64)

    template_transformer = caffe.io.Transformer({'data': net.blobs['data_A'].data.shape})
    template_transformer.set_transpose('data', (2, 0, 1))
    search_transformer = caffe.io.Transformer({'data': net.blobs['data_P'].data.shape})
    search_transformer.set_transpose('data', (2, 0, 1))

    true_num = 0
    for i in range(test_num):
        template_path = file_list[i][0]
        search_path = file_list[i][1]
        template_filename = os.path.split(template_path)[1]
        search_filename = os.path.split(search_path)[1]
        template_img = cu.loadImageByCaffe(template_path, template_transformer, color=False)
        search_img = cu.loadImageByCaffe(search_path, search_transformer, color=False)

        net.blobs['data_A'].data[...] = template_img
        net.blobs['data_P'].data[...] = search_img
        output = net.forward()
        softmax_prob = output['softmax'][0]
        confidence = 0
        if softmax_prob[0] < softmax_prob[1]:
            confidence = 1
        result = 0
        if confidence == label_list[i][0]:
            result = 1
        true_num += result
        print '[', i, ']', template_filename, ' match ', search_filename, ': ', result
        prob = "{0:.4f}".format(float(softmax_prob[1]))
        template_origin_img = caffe.io.load_image(template_path)
        search_origin_img = caffe.io.load_image(search_path)
        save_path = template_filename + search_filename
        save_path = os.path.join(args.output_folder, save_path)
        cu.visMatchNetPair(template_origin_img, search_origin_img, prob, save_path)

    print "The percentage of true num is ", float(true_num) / test_num




