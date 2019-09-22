import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


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


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test, name):
	if name == 'IP':
		target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
						,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
						'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
						'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
						'Stone-Steel-Towers']
	elif name == 'SA':
		target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
						'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
						'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
						'Vinyard_untrained','Vinyard_vertical_trellis']
	elif name == 'PU':
		target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
						'Self-Blocking Bricks','Shadows']

	else:
		target_names = [str(a) for a in range(15)]

	classification = classification_report(y_test, y_pred, target_names=target_names)
	oa = accuracy_score(y_test, y_pred)
	confusion = confusion_matrix(y_test, y_pred)
	each_acc, aa = AA_andEachClassAccuracy(confusion)
	kappa = cohen_kappa_score(y_test, y_pred)

	return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
