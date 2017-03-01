import cv2
import argparse
import os


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Resize images')
	parser.add_argument('input_image_list', help='input image list')
	parser.add_argument('output_folder', help='output folder')
	parser.add_argument('width', type=int, help='new image width')
	parser.add_argument('height', type=int, help='new image height')
	args = parser.parse_args()

	if not os.path.isfile(args.input_image_list):
		raise IOError(('Input image list not found: {:s}.\n').format(args.input_image_list))

	width = args.width
	height = args.height

	save_file_list = os.path.join(args.output_folder, 'patch_list.txt')
	image_folder = os.path.join(args.output_folder, 'images')
	if not os.path.exists(image_folder):
		os.mkdir(image_folder)

	with open(args.input_image_list) as f_list:
		with open(save_file_list, 'w') as f_save:
			for line in f_list:
				image_path, extra = line.strip().split(' ', 1)
				img = cv2.imread(image_path)
				print 'Resize image ', image_path
				print extra
				resized_img = cv2.resize(img, (width, height))
				image_filename = os.path.basename(image_path)
				save_path = os.path.join(image_folder, image_filename)
				cv2.imwrite(save_path, resized_img)
				f_save.write(save_path + " " + extra + "\n")
