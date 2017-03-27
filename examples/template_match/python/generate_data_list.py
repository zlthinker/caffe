import h5py
import os, sys, argparse, logging, time
import numpy as np
import caffe
import skimage

from generate_HDF5_data import loadMPOImage

# override caffe.io.load to handle MPO images
def loadMPOImage(image_path, color = True):
	img = skimage.io.imread(image_path, as_grey=not color)
	if img.size > 1:
		img = img[0]
	img = skimage.img_as_float(img).astype(np.float32)
	if img.ndim == 2:
		img = img[:, :, np.newaxis]
		if color:
			img = np.tile(img, (1, 1, 3))
	elif img.shape[2] == 4:
		img = img[:, :, :3]
	return img





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Generate hdf5 data for CNN')
	parser.add_argument('input', help = 'input file, each line format: [img1 path] [img2 path] [label vector]')
	parser.add_argument('output_folder', help = 'output folder')

	logging.getLogger().setLevel(logging.INFO)
	args = parser.parse_args()
	if not os.path.exists(args.input):
		logging.error('input file not exists: %s', args.input)

	if not os.path.exists(args.output_folder):
		os.mkdir(args.output_folder)
	args.output_file = os.path.join(args.output_folder, 'template_match_data.h5')

	time_start = time.time()
	generateHDF5FromText(args.input, args.output_file, 7)
	print '-------------------------------------------------------'
	logging.info('Duration of HDF5 file generation is %d sec.', time.time() - time_start)
	print '-------------------------------------------------------'



