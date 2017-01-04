import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import logging

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

caffe_root = '../'
GPU_mode = True
image_path1 = caffe_root + 'examples/images/0001.JPG'
image_path2 = caffe_root + 'examples/images/cat_gray.jpg'

# load net and set up preprocessing
if GPU_mode:
    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

img = caffe.io.load_image(image_path1)

if len(img.shape) > 4:
    img_new = img[0, :, :, :]
    logging.warning('It\'s MPO image type. Its shape is ' + str(img.shape))
else:
    img_new = img

transformed_image = transformer.preprocess('data', img_new)
# plt.imshow(img_new)
# image_path, ext = os.path.splitext(image_path1)
# save_path = image_path + '-1' + ext
# print 'save path: ', save_path
# plt.imsave(save_path, img_new)
# exit()


# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

# perform classification
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    print 'label file not exists: ', labels_file
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label:', labels[output_prob.argmax()]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])

for layer_name, blob in net.blobs.iteritems():
    print layer_name, '\t', str(blob.data.shape)

