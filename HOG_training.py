import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import argparse
import glob
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


class HOG_Trainer(object):
	def __init__(self, train_descriptors, test_descriptors):
		self.train_descriptors = train_descriptors
		self.test_descriptors = test_descriptors

	def train_HOG(self, categories):
		clf = LinearSVC(penalty='l2', random_state=1, tol=1e-3)
		clf.fit(train_descriptors, categories)
		return clf


if __name__ == '__main__':

	#Loading in the training data
	train_folders = glob.glob(os.getcwd() + '/train/*')
	classes = [f for f in os.listdir(os.getcwd() + '/train/') if not f.startswith('.')]
	le = preprocessing.LabelEncoder()
	categories = le.fit(classes)

	train_descriptors = np.load('train_descriptor.npy')
	test_descriptors = np.load('test_descriptor.npy')

	train_labels = np.load('train_labels.npy')
	test_labels = np.load('test_labels.npy')

	#Train a new classifier 
	new_hog_trainer = HOG_Trainer(train_descriptors, test_descriptors)
	clf = new_hog_trainer.train_HOG(train_labels)
	test_predictions = clf.predict(test_descriptors)

	#Obtain the prediction and the confusion matrix
	final_confusion_matrix = confusion_matrix(test_labels, test_predictions)
	test_predictions = le.inverse_transform(test_predictions)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(final_confusion_matrix)
	plt.title('Confusion matrix of the SVM classifier with histogram of oriented gradients')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + classes)
	ax.set_yticklabels([''] + classes)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()
