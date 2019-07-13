import sys
import numpy as np
import cv2
#from matplotlib import pyplot as plt
import math
import os
#from __future__ import print_function
import argparse
import glob

#Set the number of bins
current_dir = os.getcwd()
num_of_bins = 4
bw = 4
bh = 4

#Computing the descriptor vector
class ComputeDescriptors(object):
	def __init__(self, num_of_bins):
		self.num_of_bins = num_of_bins

	def compute_descriptor_vector(self, image, bw, bh):
		width, height = image.shape[0], image.shape[1]
		deltaw = int(width/(bw + 1))
		deltah = int(height/(bh + 1))
		list_of_hist = []
		for m in range(bw):
			for n in range(bh):
				subblock = image[m*deltaw:m*deltaw + 2*deltaw, n*deltah:n*deltah + 2*deltah, :]
				subblock = subblock.reshape(subblock.shape[0]*subblock.shape[1], 3)
				hist, _ = np.histogramdd(subblock, (self.num_of_bins, self.num_of_bins, self.num_of_bins))
				hist = np.ravel(hist)
				list_of_hist.append(hist)
		descriptor = np.concatenate(list_of_hist, axis=None)
		return descriptor

if __name__ == '__main__':

	train_dir = current_dir + '/train'
	test_dir = current_dir + '/test'

	descriptor_maker = ComputeDescriptors(num_of_bins)

	train_descriptor_list = []
	test_descriptor_list = []

	#Compute the descriptor for all images in the folder
	for folder_name in glob.glob(train_dir + '/*'):
		for image_name in glob.glob(folder_name + '/*'):
			image = cv2.imread(image_name, 1)
			train_descriptor = descriptor_maker.compute_descriptor_vector(image, bw, bh)
			train_descriptor_list.append(train_descriptor)
	
	#Put the train descriptors into a numpy array and save them
	train_descriptor_block = np.concatenate(train_descriptor_list, axis=0)
	train_descriptor_block = train_descriptor_block.reshape(4000, -1)
	print(train_descriptor_block.shape)
	np.save('train_descriptor.npy', train_descriptor_block)

	for folder_name in glob.glob(test_dir + '/*'):
		for image_name in glob.glob(folder_name + '/*'):
			image = cv2.imread(image_name, 1)
			test_descriptor = descriptor_maker.compute_descriptor_vector(image, bw, bh)
			test_descriptor_list.append(test_descriptor)
	
	#Put the test descriptors into a numpy array and save them
	test_descriptor_block = np.concatenate(test_descriptor_list, axis=0)
	test_descriptor_block = test_descriptor_block.reshape(1000, -1)
	print(test_descriptor_block.shape)
	np.save('test_descriptor.npy', test_descriptor_block)



















