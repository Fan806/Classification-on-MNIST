import os
import pickle
import sys

import numpy as np
import PIL.Image
import torch

from torch.utils import data
from torchvision.datasets.folder import default_loader

import pandas as pd
from torchvision.datasets.utils import download_url

class CUB(data.Dataset):
	"""CUB200-2011
	Args:
		_root, str: Root directory of the dataset.
		_train, bool: Load/test data.
		_transform, callable: A function/transform that takes in a PIL.Image 
			and transforms it.
		_target_transform, callable: A function/transform that takes in the
			target and transforms it.
		_train_data: list of np.ndarray
		_train_labels: list of int
		_test_data: list of np.ndarray
		_test_labels: list of int
	"""

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		"""Load the dataset
		Args:
			root, str: Root directory of the dataset
			train, bool [True]: Load train/test data
			tranform, callable [None]: A function/transform that takes in a
				PIL.Image and transforms it
			target_transform, cllable [None]: A function/transform that takes
				in the target and transforms it.
			download, bool [False]: If true, downloads the dataset from the 
				internet and puts it in root directory. If dataset is already 
				downloaded, it is not download again.
		"""
		self._root = os.path.expanduser(root)
		self._train = train
		self._transform = transform
		self._target_transform = target_transform

		if not os.path.isdir(self._root):
			os.mkdir(self._root)

		if self._checkIntegrity():
			print("Files already downloaded and verified.")
		elif download:
			url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
			self._download(url)
			self._extract()
		else:
			raise RuntimeError('Dataset not found. You can use download=True to download it.')

		# load the picked data
		if self._train:
			self._train_data, self._train_labels = pickle.load(open(
				os.path.join(self._root, 'processed/train.pkl'), 'rb'))
		else:
			self._test_data, self._test_labels = pickle.load(open(
				os.path.join(self._root, 'processed/test.pkl'), 'rb'))

	def __getitem__(self, index):
		"""
		Args:
			index, int: Index
		Returns:
			image, PIL.Image: Image of the given index
			target, str: target of the given index
		"""
		if self._train:
			image, target = self._train_data[index], self._train_labels[index]
		else:
			image, target = self._test_data[index], self._test_labels[index]

		image = PIL.Image.formarray(image)

		if self._transform is not None:
			image = self._transform(image)
		if self._target_transform is not None:
			target = self._target_transform(target)

		return image, target

	def __len__(self):
		"""length of the dataset
		Returns:
			length, int: Length of the dataset
		"""
		if self._train:
			return len(self._train_data)
		return len(self._test_data)

	def _checkIntegrity(self):
		"""Check whether we have already processed the data
		Returns:
			flag, bool: True is  we have already processed the data
		"""
		return (
			os.path.isfile(os.path.join(self._root, 'processed/train.pkl'))
			and os.path.isfile(os.path.join(self._root, 'processed/test.pkl')))

	def _download(self, url):
		"""Download and uncompress the tar.gz file from a given URL
		Args:
			url, str: URL to be downloaded.
		"""
		import six.moves
		import tarfile

		raw_path = os.path.join(self._root, 'raw')
		processed_path = os.path.join(self._root, 'processed')
		if not os.path.isdir(raw_path):
			os.mkdir(raw_path)
		if not os.path.isdir(processed_path):
			os.mkdir(processed_path)

		# Download file
		fpath = os.path.join(self._root, 'raw/CUB_200_2011.tgz')

		def _recall_func(num, block_size, total_size):
			sys.stdout.write('\rdownloading %.1f%%     downloaded size: %d     total size: %d' % (float(num*block_size)/float(total_size)*100.0, num*block_size, total_size))
			sys.stdout.flush()

		if not os.path.isfile(fpath):
			try:
				print('Downloading ' + url + ' to ' + fpath)
				six.moves.urllib.request.urlretrieve(url, fpath, _recall_func)
			except six.moves.urllib.error.URLError:
				print("Failed download.")
		if not os.path.isfile(os.path.join(self._root, 'raw/CUB_200_2011')):
			# Extract file
			cwd = os.getcwd()
			tar = tarfile.open(fpath, 'r:gz')
			os.chdir(os.path.join(self._root, 'raw'))
			tar.extractall()
			tar.close()
			os.chdir(cwd)

	def _extract(self):
		"""Prepare the data for train/test split and save onto disk"""
		image_path = os.path.join(self._root, "raw/CUB_200_2011/images/")
		# Format of images.txt: <image_id> <image_name>
		id2name = np.genfromtxt(os.path.join(
			self._root, 'raw/CUB_200_2011/images.txt'), dtype=str)
		# Format of train_test_split.txt: <image_id> <is_training_image>
		id2train = np.genfromtxt(os.path.join(
			self._root, 'raw/CUB_200_2011/train_test_split.txt'), dtype=int)

		train_data = []
		train_labels = []
		test_data = []
		test_labels = []
		for id_ in range(id2name.shape[0]):
			image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
			label = int(id2name[id_, 1][:3]) - 1	# Label starts with 0

			# Convert gray scale image to RGB image
			if image.getbands()[0] == 'L':
				image = image.convert('RGB')
			image_np = np.array(image)
			image.close()

			if id2train[id_, 1] == 1:
				train_data.append(image_np)
				train_labels.append(label)
			else:
				test_data.append(image_np)
				test_labels.append(label)

		pickle.dump((train_data, train_labels),
					open(os.path.join(self._root, 'processed/train.pkl'), 'wb'))
		pickle.dump((test_data, test_labels),
					open(os.path.join(self._root, 'processed/test.pkl'), 'wb'))

# -------------------------------------------------------------------------------------------------------------------------------------------------------

class PascalVOC(data.Dataset):
	"""Pascal VOC 2012
	Args:
		_root, str: Root directory of the dataset.
		_train, bool: Load/test data.
		_transform, callable: A function/transform that takes in a PIL.Image 
			and transforms it.
		_target_transform, callable: A function/transform that takes in the
			target and transforms it.
		_train_data: list of np.ndarray
		_train_labels: list of int
		_test_data: list of np.ndarray
		_test_labels: list of int
	"""

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		"""Load the dataset
		Args:
			root, str: Root directory of the dataset
			train, bool [True]: Load train/test data
			tranform, callable [None]: A function/transform that takes in a
				PIL.Image and transforms it
			target_transform, cllable [None]: A function/transform that takes
				in the target and transforms it.
			download, bool [False]: If true, downloads the dataset from the 
				internet and puts it in root directory. If dataset is already 
				downloaded, it is not download again.
		"""
		self._root = os.path.expanduser(root)
		self._train = train
		self._transform = transform
		self._target_transform = target_transform

		if not os.path.isdir(self._root):
			os.mkdir(self._root)

		if self._checkIntegrity():
			print("Files already downloaded and verified.")
		elif download:
			url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
			self._download(url)
			self._extract()
		else:
			raise RuntimeError('Dataset not found. You can use download=True to download it.')

		# load the picked data
		if self._train:
			self._train_data, self._train_labels = pickle.load(open(
				os.path.join(self._root, 'processed/train.pkl'), 'rb'))
		else:
			self._test_data, self._test_labels = pickle.load(open(
				os.path.join(self._root, 'processed/test.pkl'), 'rb'))

	def __getitem__(self, index):
		"""
		Args:
			index, int: Index
		Returns:
			image, PIL.Image: Image of the given index
			target, str: target of the given index
		"""
		if self._train:
			image, target = self._train_data[index], self._train_labels[index]
		else:
			image, target = self._test_data[index], self._test_labels[index]

		image = PIL.Image.formarray(image)

		if self._transform is not None:
			image = self._transform(image)
		if self._target_transform is not None:
			target = self._target_transform(target)

		return image, target

	def __len__(self):
		"""length of the dataset
		Returns:
			length, int: Length of the dataset
		"""
		if self._train:
			return len(self._train_data)
		return len(self._test_data)

	def _checkIntegrity(self):
		"""Check whether we have already processed the data
		Returns:
			flag, bool: True is  we have already processed the data
		"""
		return (
			os.path.isfile(os.path.join(self._root, 'processed/train.pkl'))
			and os.path.isfile(os.path.join(self._root, 'processed/test.pkl')))

	def _download(self, url):
		"""Download and uncompress the tar.gz file from a given URL
		Args:
			url, str: URL to be downloaded.
		"""
		import six.moves
		import tarfile

		raw_path = os.path.join(self._root, 'raw')
		processed_path = os.path.join(self._root, 'processed')
		if not os.path.isdir(raw_path):
			os.mkdir(raw_path)
		if not os.path.isdir(processed_path):
			os.mkdir(processed_path)

		# Download file
		fpath = os.path.join(self._root, 'raw/PascalVOC_2012.tgz')

		def _recall_func(num, block_size, total_size):
			sys.stdout.write('\rdownloading %.1f%%     downloaded size: %d     total size: %d' % (float(num*block_size)/float(total_size)*100.0, num*block_size, total_size))
			sys.stdout.flush()

		try:
			print('Downloading ' + url + ' to ' + fpath)
			six.moves.urllib.request.urlretrieve(url, fpath, _recall_func)
		except six.moves.urllib.error.URLError:
			print("Failed download.")

		# Extract file
		cwd = os.getcwd()
		tar = tarfile.open(fpath, 'r:gz')
		os.chdir(os.path.join(self._root, 'raw'))
		tar.extractall()
		tar.close()
		os.chdir(cwd)

	def _extract(self):
		"""Prepare the data for train/test split and save onto disk"""
		image_path = os.path.join(self._root, "raw/PascalVOC_2012/images/")
		# Format of images.txt: <image_id> <image_name>
		id2name = np.genfromtxt(os.path.join(
			self._root, 'raw/PascalVOC_2012/images.txt'), dtype=str)
		# Format of train_test_split.txt: <image_id> <is_training_image>
		id2train = np.genfromtxt(os.path.join(
			self._root, 'raw/PascalVOC_2012/train_test_split.txt'), dtype=int)

		train_data = []
		train_labels = []
		test_data = []
		test_labels = []
		for id_ in range(id2name.shape[0]):
			image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
			label = int(id2name[id_, 1][:3]) - 1	# Label starts with 0

			# Convert gray scale image to RGB image
			if image.getbands()[0] == 'L':
				image = image.convert('RGB')
			image_np = np.array(image)
			image.close()

			if id2train[id_, 1] == 1:
				train_data.append(image_np)
				train_labels.append(label)
			else:
				test_data.append(image_np)
				test_labels.append(label)

		pickle.dump((train_data, train_labels),
					open(os.path.join(self._root, 'processed/train.pkl'), 'wb'))
		pickle.dump((test_data, test_labels),
					open(os.path.join(self._root, 'processed/test.pkl'), 'wb'))


