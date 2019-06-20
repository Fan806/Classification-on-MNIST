import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from netmodel import *
from CustomDataset import CUB

import numpy as np

def Cluster(data, num_cluster):
	kmeans_model = KMeans(n_clusters=num_cluster).fit(data)
	labels = kmeans_model.labels_
	return labels

def PCAReduction(visualize, num_cluster, num_components):
	pca = PCA(n_components=num_components)
	pca_results = pca.fit_transform(visualize)
	labels = Cluster(pca_results, num_cluster)
	return pca_results, labels

def TSNEVisualization(results, labels, num_cluster, title, name):
	tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(results)

	color = sns.color_palette("hls", num_cluster)

	fig = plt.figure()
	axes = fig.add_subplot(111)
	legends = {}
	label_legends = np.arange(num_cluster)
	for i in range(tsne_results.shape[0]):
		legends[labels[i]] = axes.scatter(tsne_results[i,0],tsne_results[i,1],color=color[labels[i]],alpha=0.5)

	# tmp = axes.scatter(tsne_results[:,0],tsne_results[:,1],c=np.sqrt((np.square(tsne_results[:,1]))+np.square(tsne_results[:,0])),cmap=cmap, alpha=0.75)
	# fig.colorbar(tmp, shrink=0.6)

	plt.legend(legends.values(),label_legends)
	plt.title(title)
	plt.savefig(name)

def train(isload, net, train_loader, in_dim, inter_dim, lr, epochs, num_cluster, num_components, path):
	# isGPU = torch.cuda.is_available()
	isGPU = False
	print("isGPU: ",isGPU)

	if isload=='y':
		model = torch.load(path)
	else:
		if net=="AlexNet":
			model = AlexNet(in_dim, inter_dim, num_cluster)
		if net=="VGG16":
			model = vgg16_bn(in_dim=in_dim, num_classes=num_cluster)
		if net=="ResNet":
			model = ResNet18(in_dim=in_dim, num_classes=num_cluster)
	if isGPU:
		torch.cuda.empty_cache()
		model = model.cuda()
		
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
	with open('withdecay'+'1.txt','w') as f:
		for epoch in range(epochs):
			running_acc = 0.0
			running_loss = 0.0
			visualize = []
			scheduler.step()
			for step, data in enumerate(train_loader, 1):
				img, label = data
				img = Variable(img)
				label = Variable(label)
				if isGPU:
					img = img.cuda()
					label = label.cuda()

				# forward
				out, inter_layer = model(img)
				visualize.append(inter_layer.data.numpy())
				# print("main inter_layer: ",inter_layer.shape)
				loss = criterion(out, label)

				# backward
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				_, pred = torch.max(out, dim=1)

				# accumulate loss
				running_loss += loss.item()*label.size(0)

				# accumulate the number of correct samples
				current_num = (pred==label).sum()

				acc = (pred == label).float().mean()
				running_acc += current_num.item()

				if step%50 == 0:
					# print("draw picture")
					# break
					# torch.save(model, path+"%d_%d"%(step,epoch+1))
					f.write("{:.6f}\n".format(loss.item()))
					print("epoch: {}/{}, loss: {:.6f}, running_acc: {:.6f}".format(epoch+1, epochs, loss.item(), acc.item()))
		print("epoch: {}, loss: {:.6f}, accuracy: {:.6f}".format(epoch+1, running_loss, running_acc/len(train_loader.dataset)))

	dim1, dim2 = visualize[0].shape
	length = len(visualize)
	visualize = np.array(visualize).reshape((length*dim1, dim2))
	# visualization
	title1 = "Visualization of the intermediate-layer"
	title2 = "Visualization of the PCA feature"
	labels = Cluster(visualize, num_cluster)
	TSNEVisualization(visualize, labels, num_cluster, title1,'SNE_Cluster.png')
	pca_results, labels = PCAReduction(visualize, num_cluster, num_components)
	TSNEVisualization(pca_results, labels, num_cluster, title2,'SNE_PCA.png')

def test(test_loader, num_cluster, path):
	# isGPU = torch.cuda.is_available()
	isGPU = False
	model = torch.load(path)
	model.eval()
	current_num = 0
	for i, data in enumerate(test_loader, 1):
		img, label = data
		if isGPU:
			img = img.cuda()
			label = label.cuda()
		with torch.no_grad():
			img = Variable(img)
			label = Variable(label)
		out, inter_layer = model(img)
		_, pred = torch.max(out, 1)
		current_num += (pred == label).sum().item()
	print("accuracy: {:.6f}".format(current_num/len(test_loader.dataset)))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-path', required=True)
	parser.add_argument('-load',choices=['y','n'])
	parser.add_argument('-dataset', choices=['MNIST','CUB','CIFAR'], required=True)
	parser.add_argument('-pattern', choices=['train','test'], required=True)
	parser.add_argument('-net', choices=['AlexNet','VGG16','VGG19','ResNet'],required=True)
	args = parser.parse_args()

	batch_size = 60
	lr = 0.001
	epochs = 4

	num_cluster = 1

	data_transform = transforms.Compose([
		transforms.Resize(256),
		# transforms.Resize(224),
		transforms.RandomCrop(224), 
		transforms.ToTensor(),
		transforms.Normalize((0.1307,),(0.3081,))
		]
		)

	if args.pattern == 'train':
		if args.dataset == 'MNIST':
			X_train = datasets.MNIST('./datasets/MNIST', train=True, transform=data_transform, download=True)
			num_cluster = 10
			in_dim = 1
			inter_dim = 2
			num_components = 8
		elif args.dataset == 'CIFAR':
			X_train = datasets.CIFAR10('./datasets/CIFAR', train=True, transform=data_transform, download=True)
			num_cluster = 10
			in_dim = 3
			inter_dim = 3
			num_components = 8
		elif args.dataset == 'CUB':
			X_train = CUB('./datasets/CUB', train=True, transform=data_transform, download=False)
			num_cluster = 200
			in_dim = 3
			inter_dim = 3
			num_components = 100

		train_loader = DataLoader(X_train, batch_size, shuffle=True)
		train(args.load, args.net,train_loader, in_dim, inter_dim, lr, epochs, num_cluster, num_components, args.path)
	else:
		if args.dataset == 'MNIST':
			X_test = datasets.MNIST('./datasets/MNIST', train=False, transform=data_transform, download=True)
			num_cluster = 10
		elif args.dataset == 'CIFAR':
			X_test = datasets.CIFAR10('./datasets/CIFAR', train=False, transform=transforms.ToTensor(), download=True)
			num_cluster = 10
		elif args.dataset == 'CUB':
			X_test = CUB('./datasets/CUB', train=False, transform=transforms.ToTensor(), download=True)
			num_cluster = 10

		test_loader = DataLoader(X_test)
		test(test_loader, num_cluster, args.path)


if __name__ == '__main__':
	main()
	





