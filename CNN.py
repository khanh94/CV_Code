import sys
import numpy as np
import cv2
#from matplotlib import pyplot as plt
import math
import os
#from __future__ import print_function
import argparse
from sklearn.svm import LinearSVC
import torch 
import glob
import torch
import torch.nn as nn
import torch.nn.functional  as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Set device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Set Hyper parameters
num_epochs = 20
num_classes = 5
batch_size = 10
dropout_rate = 0.6
learning_rate = 0.00008

class ImageLoader(Dataset):
	def __init__(self, x, y, isCuda=False):
		self.X = x
		self.y = y
		
	def __getitem__(self, index):
		x_val = self.X[index]
		x_val = x_val/255	
		x_val = torch.from_numpy(x_val).permute(2, 1, 0)
		x_val = x_val.type(torch.FloatTensor)
		y_val = torch.from_numpy(np.array(self.y[index]))
		return x_val, y_val

	def __len__(self):
		return len(self.X)

class CNNClassifier(torch.nn.Module):
	def __init__(self, num_classes=2):
		super(CNNClassifier, self).__init__()

		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
		self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
		self.fc1 = nn.Linear(in_features = 8*8*128, out_features = 128)
		self.fc2 = nn.Linear(in_features = 128, out_features = num_classes)       
		self.dropout_rate = dropout_rate

	def forward(self, image):
		out = self.conv1(image)      # batch_size x 32 x 64 x 64
		out = F.relu(F.max_pool2d(out, 2))     # batch_size x 32 x 32 x 32
		out = self.conv2(out)        # batch_size x 64 x 32 x 32
		out = F.relu(F.max_pool2d(out, 2))     # batch_size x 64 x 16 x 16
		out = self.conv3(out)        # batch_size x 128 x 16 x 16
		out = F.relu(F.max_pool2d(out, 2))     # batch_size x 128 x 8 x 8

		out = out.view(-1, 8*8*128)  # batch_size x 8*8*128
		# apply 2 fully connected layers with dropout
		out = F.dropout(F.relu(self.fc1(out)), p=self.dropout_rate) #training=self.training)# batch_size x 128
		out = self.fc2(out) # batch_size x 5

		return F.log_softmax(out, dim=1)
		

if __name__ == '__main__':

	#Loading in the training data
	train_folders = glob.glob(os.getcwd() + '/train/*')
	classes= [f for f in os.listdir(os.getcwd() + '/train/') if not f.startswith('.')]
	print(classes)
	le = preprocessing.LabelEncoder()
	categories = le.fit_transform(classes)

	train_image_names = []
	train_category_count = []

	for folder in train_folders:
		for f in glob.glob(folder + '/*.JPEG'):
			train_image_names.append(f)

		train_category_count.append(len(os.listdir(folder)))

	train_labels = np.array([k for k,v in zip(categories, train_category_count) for _ in range(v)])
	print(train_labels)
	np.save('train_labels.npy', train_labels)
	train_images = np.array([cv2.resize(cv2.imread(image), (64, 64)) for image in train_image_names])
	print(train_images.shape)

	#Loading in the testing data
	test_folders = glob.glob(os.getcwd() + '/test/*')

	test_image_names = []
	test_category_count = []

	for folder in test_folders:
		for f in glob.glob(folder + '/*.JPEG'):
			test_image_names.append(f)

		test_category_count.append(len(os.listdir(folder)))

	test_labels = np.array([k for k,v in zip(categories, test_category_count) for _ in range(v)])
	print(test_labels.shape)
	np.save('test_labels.npy', test_labels)
	test_images = np.array([cv2.resize(cv2.imread(image), (64, 64)) for image in test_image_names])

	print(test_images.shape)
	use_cuda = torch.cuda.is_available()

	train_img_loader = ImageLoader(train_images, train_labels, use_cuda)
	train_loader = DataLoader(train_img_loader, batch_size=batch_size, shuffle=False, num_workers=1)

	test_img_loader = ImageLoader(test_images, test_labels, use_cuda)
	test_loader = DataLoader(test_img_loader, batch_size=batch_size, shuffle=False, num_workers=1)

	model = CNNClassifier(num_classes).to(device)

	#Set up the Loss function and the Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	#Train the model
	total_step = len(train_loader)
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = images.to(device)
			labels = labels.to(device)

			#Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)

			#Backward pass and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i + 1) % 200 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i+1, total_step, loss.item()))

	#Test the model
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0

		list_of_predictions = []
		for images, labels in test_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			list_of_predictions.append(predicted)
			print(predicted)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()


		test_predictions = torch.stack(list_of_predictions)
		test_predictions = test_predictions.view(1000)
		print(test_predictions)
		print('Test accuracy of the model on the 1000 test images: {} %'.format(100*correct/total))
	
	#Save model checkpoint
	torch.save(model.state_dict(), 'model.ckpt')

	test_predictions = test_predictions.numpy()
	final_confusion_matrix = confusion_matrix(test_labels, test_predictions)
	test_predictions = le.inverse_transform(test_predictions)

	fig = plt.figure()
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plot_confusion_matrix(final_confusion_matrix, classes=classes,
                      title='Confusion matrix of the CNN classifier')

	plt.show()