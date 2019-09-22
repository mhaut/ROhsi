import argparse
import auxil

from hyper_pytorch import *

import models

import torch
import torch.nn.parallel
from torchvision.transforms import *
import transform






def load_hyper(args):
	data, labels, numclass = auxil.loadData(args.dataset, num_components=args.components)
	pixels, labels = auxil.createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels = True)
	bands = pixels.shape[-1]; numberofclass = len(np.unique(labels))
	x_train, x_test, y_train, y_test = auxil.split_data(pixels, labels, args.tr_percent)
	del pixels, labels
	if args.p != 0: transform_train = transform.RandomErasing(probability = args.p, sh = args.sh, r1 = args.r1, )
	else: transform_train = None
	train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train), transform_train)
	test_hyper  = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test), None)
	kwargs = {'num_workers': 1, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
	test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
	return train_loader, test_loader, numberofclass, bands


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
	model.train()
	accs   = np.ones((len(trainloader))) * -1000.0
	losses = np.ones((len(trainloader))) * -1000.0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		losses[batch_idx] = loss.item()
		accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	return (np.average(losses), np.average(accs))


def test(testloader, model, criterion, epoch, use_cuda):
	model.eval()
	accs   = np.ones((len(testloader))) * -1000.0
	losses = np.ones((len(testloader))) * -1000.0
	for batch_idx, (inputs, targets) in enumerate(testloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
		outputs = model(inputs)
		losses[batch_idx] = criterion(outputs, targets).item()
		accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
	return (np.average(losses), np.average(accs))


def predict(testloader, model, criterion, use_cuda):
	model.eval()
	predicted = []
	for batch_idx, (inputs, targets) in enumerate(testloader):
		if use_cuda: inputs = inputs.cuda()
		inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
		[predicted.append(a) for a in model(inputs).data.cpu().numpy()] 
	return np.array(predicted)






def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='IP', type=str, help='the path to your dataset')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate of adam optimizer')
	parser.add_argument('--epochs', default=600, type=int, help='number of total epochs')
	parser.add_argument('--batch_size', default=100, type=int, help='batch size')
	parser.add_argument('--components', default=1, type=int, help='dimensionality reduction')
	parser.add_argument('--spatialsize', default=23, type=int, help='windows size')
	parser.add_argument('--tr_percent', default=0.10, type=float, metavar='N', 
					 help='train set size')
	parser.add_argument('--tr_bsize', default=100, type=int, metavar='N',
						help='train batchsize')
	parser.add_argument('--te_bsize', default=5000, type=int, metavar='N',
						help='test batchsize')

	parser.add_argument('--p', default=0, type=float, help='Random Erasing probability')
	parser.add_argument('--sh', default=0.3, type=float, help='max erasing area')
	parser.add_argument('--r1', default=0.2, type=float, help='aspect of erasing area')
	
	# ALERT: cambiar esto
	parser.add_argument("--verbose", action='store_true', help="Verbose? Default NO")

	args = parser.parse_args()
	state = {k: v for k, v in args._get_kwargs()}

	trainloader, testloader, num_classes, bands = load_hyper(args)

	# Use CUDA
	use_cuda = torch.cuda.is_available()
	if use_cuda: torch.backends.cudnn.benchmark = True

	model = models.SimpleCNN(bands, args.spatialsize, num_classes)
	model = torch.nn.DataParallel(model)
	if use_cuda: model = model.cuda()

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())


	title = 'HYPER-' + args.dataset

	best_acc = -1
	for epoch in range(args.epochs):
		#adjust_learning_rate(optimizer, epoch)
		if args.verbose: print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

		train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
		test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

		print("EPOCH", epoch, "TRAIN LOSS", test_loss, "TRAIN ACCURACY", test_acc, end=',')
		print("LOSS", test_loss, "ACCURACY", test_acc)
		# save model
		if test_acc > best_acc:
			state = {
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'acc': test_acc,
					'best_acc': best_acc,
					'optimizer' : optimizer.state_dict(),
			}
			torch.save(state, "best_model"+str(args.p)+".pth.tar")
			best_acc = test_acc

	checkpoint = torch.load("best_model"+str(args.p)+".pth.tar")
	best_acc = checkpoint['best_acc']
	start_epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
	print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)
	classification, confusion, results = auxil.reports(np.argmax(predict(testloader, model, criterion, use_cuda), axis=1), np.array(testloader.dataset.__labels__()), args.dataset)
	print(args.dataset, results)
	

if __name__ == '__main__':
	main()
