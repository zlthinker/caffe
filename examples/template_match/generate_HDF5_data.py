import h5py
import os, sys, argparse, logging, time
import numpy as np
import caffe
import skimage

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


# format of input file each line: [img1 path] [img2 path] [label vector]
def generateHDF5FromText(input_file, output_file, label_dim):
	logging.info('generating hdf5 file from text...')

	img_paths0 = []
	img_paths1 = []
	confidence_vals = []
	regression_vals = []
	with open(input_file) as f_in:
		for line in f_in:
			val_list = line.strip().split('; ')
			if len(val_list) != label_dim + 2:
				continue
			img_path0 = val_list[0]
			img_path1 = val_list[1]
			if not (os.path.exists(img_path0) and os.path.exists(img_path1)):
				continue
			confidence_val = float(val_list[2])
			regression_val = []
			for i in range(3, label_dim + 2):
				regression_val.append(float(val_list[i]))
			img_paths0.append(img_path0)
			img_paths1.append(img_path1)
			confidence_vals.append(confidence_val)
			regression_vals.append(regression_val)

	# some global variables
	MAX_MEM = 8 * 1024 * 1024 * 1024 # max memory allocated = 16GB
	DATUM_SIZE = (256*256 + 32*32) * 3 * 4 + label_dim * 4 # in bytes
	MAX_DATUM_NUM = 1000#MAX_MEM / DATUM_SIZE
	DATUM_NUM = len(img_paths0)
	CHUNK_NUM = DATUM_NUM / MAX_DATUM_NUM
	if not DATUM_NUM % MAX_DATUM_NUM == 0:
		CHUNK_NUM = CHUNK_NUM + 1
	
	f_out = h5py.File(output_file, 'w')
	data0_set = f_out.create_dataset("data0", shape=(1, 3, 32, 32), maxshape=(None, 3, 32, 32), dtype="f4")
	data1_set = f_out.create_dataset("data1", shape=(1, 3, 256, 256), maxshape=(None, 3, 256, 256), dtype="f4")
	confidence_set = f_out.create_dataset('confidence', shape=(1, 1), maxshape=(None, 1), dtype="f4")
	regression_set = f_out.create_dataset("regression", shape=(1, label_dim - 1), maxshape=(None, label_dim - 1), dtype="f4")

	logging.info('Total %d datums, split into %d chunks', DATUM_NUM, CHUNK_NUM)
	# write hdf5 file by chunks
	for c in range(CHUNK_NUM):
		start_id = c * MAX_DATUM_NUM
		chunk_size = MAX_DATUM_NUM
		if c == CHUNK_NUM - 1:
			chunk_size = DATUM_NUM - (CHUNK_NUM - 1) * MAX_DATUM_NUM
		logging.info("Writing chunk %d, chunk size is %d", c, chunk_size)
		datas0 = np.zeros((chunk_size, 3, 32, 32), dtype='f4') # 32-bit floating number
		datas1 = np.zeros((chunk_size, 3, 256, 256), dtype='f4') 
		confidence_labels = np.zeros((chunk_size, 1), dtype="f4")
		regression_labels = np.zeros((chunk_size, label_dim - 1), dtype="f4")
		# resize datasets
		data0_set.resize((MAX_DATUM_NUM * c + chunk_size, 3, 32, 32))
		data1_set.resize((MAX_DATUM_NUM * c + chunk_size, 3, 256, 256))
		confidence_set.resize((MAX_DATUM_NUM * c + chunk_size, 1))
		regression_set.resize((MAX_DATUM_NUM * c + chunk_size, label_dim - 1))

		for i in range(chunk_size):
			id = i + start_id
			if id % 1000 == 0:
				logging.info("Finish %d.", id)
			logging.debug('[%d] read image: %s', id, img_paths0[id])
			img0 = loadMPOImage( img_paths0[id] )
			img0 = caffe.io.resize( img0, (32, 32, 3) ) # resize to fixed patch size
			img0 = np.transpose( img0 , (2,0,1)) # switch dimension, channel first
			datas0[i] = img0
			img1 = loadMPOImage( img_paths1[id] )
			img1 = caffe.io.resize( img1, (256, 256, 3) ) # resize to fixed patch size
			img1 = np.transpose( img1 , (2,0,1)) # switch dimension, channel first
			datas1[i] = img1
			confidence_labels[i] = confidence_vals[id]
			regression_labels[i] = np.asarray(regression_vals[id])

		data0_set[start_id : start_id + chunk_size] = datas0
		data1_set[start_id : start_id + chunk_size] = datas1
		confidence_set[start_id : start_id + chunk_size] = confidence_labels
		regression_set[start_id : start_id + chunk_size] = regression_labels

	f_out.close()

	# debug info
	file = h5py.File(output_file, 'r')
	keys = file.keys()
	for key in keys:
		dataset = file[key]
		print "-----------------"
		print 'key: ', key
		print 'shape', dataset.shape



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



