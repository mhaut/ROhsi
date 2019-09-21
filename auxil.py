import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def shuffle_unison(a, b):
	assert len(a) == len(b)
	p = numpy.random.permutation(len(a))
	return a[p], b[p]

def split_data(pixels, labels, percent, splitdset="custom"):
	splitdset = "sklearn"
	if splitdset == "sklearn":
		return train_test_split(pixels, labels, test_size=(1-percent), stratify=labels, random_state=42)
	elif splitdset == "custom":
		# ALERT: ADD CODE
		print("NOT WORK")
		exit()



def loadData(name, num_components=None):
	data_path = os.path.join(os.getcwd(),'data')
	if name == 'IP':
		data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
		labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
	elif name == 'SA':
		data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
		labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
	elif name == 'PU':
		data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
		labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
	else:
		print("NO DATASET")
		exit()

	shapeor = data.shape
	data = data.reshape(-1, data.shape[-1])
	if num_components != None:
		data = PCA(n_components=num_components).fit_transform(data)
		shapeor = np.array(shapeor)
		shapeor[-1] = num_components
	data = MinMaxScaler().fit_transform(data)
	data = data.reshape(shapeor)
	num_class = len(np.unique(labels)) - 1
	return data, labels, num_class


def padWithZeros(X, margin=2):
	newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
	x_offset = margin
	y_offset = margin
	newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
	return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
	margin = int((windowSize - 1) / 2)
	zeroPaddedX = padWithZeros(X, margin=margin)
	# split patches
	patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
	patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
	patchIndex = 0
	for r in range(margin, zeroPaddedX.shape[0] - margin):
		for c in range(margin, zeroPaddedX.shape[1] - margin):
			patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
			patchesData[patchIndex, :, :, :] = patch
			patchesLabels[patchIndex] = y[r-margin, c-margin]
			patchIndex = patchIndex + 1
	if removeZeroLabels:
		patchesData = patchesData[patchesLabels>0,:,:,:]
		patchesLabels = patchesLabels[patchesLabels>0]
		patchesLabels -= 1
	return patchesData, patchesLabels


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res
