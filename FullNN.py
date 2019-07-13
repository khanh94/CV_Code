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
import itertools

#Set device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Set Hyper parameters
num_epochs = 200
num_classes = 5
batch_size = 20
dropout_rate = 0.4
learning_rate = 0.00008

#Build the image loader
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

#Build a Fully Connected Neural Network 
class FullyConnectedNN(torch.nn.Module):
	def __init__(self, num_classes=2):
		super(FullyConnectedNN, self).__init__()
		self.fc1 = nn.Linear(in_features = 12*18*3, out_features=24)
		self.fc2 = nn.Linear(in_features = 24, out_features=24)
		self.fc3 = nn.Linear(in_features = 24, out_features=num_classes)
		self.dropout_rate = dropout_rate

	def forward(self, image):
		out = image.view(-1, 12*18*3)
		out = F.dropout(F.relu(self.fc1(out)), p=self.dropout_rate)
		out = F.dropout(F.relu(self.fc2(out)), p=self.dropout_rate)
		out = self.fc3(out)

		return F.log_softmax(out, dim=1)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

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
	train_images = np.array([cv2.resize(cv2.imread(image), (12, 18)) for image in train_image_names])
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
	np.save('test_labels.npy', test_labels)
	test_images = np.array([cv2.resize(cv2.imread(image), (12, 18)) for image in test_image_names])

	use_cuda = torch.cuda.is_available()

	train_img_loader = ImageLoader(train_images, train_labels, use_cuda)
	train_loader = DataLoader(train_img_loader, batch_size=batch_size, shuffle=False, num_workers=1)

	test_img_loader = ImageLoader(test_images, test_labels, use_cuda)
	test_loader = DataLoader(test_img_loader, batch_size=batch_size, shuffle=False, num_workers=1)

	model = FullyConnectedNN(num_classes).to(device)

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
                      title='Confusion matrix of the Fully Connected classifier')

	plt.show()