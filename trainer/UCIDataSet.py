import torch
import os
import numpy as np


class UCIDataSet(torch.utils.data.Dataset):
  def __init__(self, labels, data):
    self.labels = labels
    self.data = data

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    label = self.labels[idx]
    vector = torch.FloatTensor(self.data[idx])
    return vector, label


def get_data_from_file(file_path, desired_gas):
  with open(file_path) as fd:
    samples = fd.read()
    samples = samples.split("\n")

  data = []
  labels = []
  for sample in samples[:-1]:
    gas, vector = sample.split(";")
    if gas is not desired_gas:
      continue

    vector = vector.split(" ")
    label = float(vector[0])
    labels.append(label)

    vector = vector[1:-1]
    vector = [float(elt.split(":")[1]) for elt in vector]
    data.append(vector)
  
  return data, labels


def get_UCI_data_loader(data_dir_path, label):
	files = os.listdir(data_dir_path)

	data = []
	labels = []
	for file in files[::1]:
	  file_path = os.path.join(data_dir_path, file)
	  data_from_file, labels_from_file = get_data_from_file(file_path, label)
	  data.extend(data_from_file)
	  labels.extend(labels_from_file)

	# Normalizing labels:
	norm_labels = np.asarray(labels)
	norm_labels -= np.mean(norm_labels, axis=0, keepdims=True)
	norm_labels /= np.std(norm_labels, axis=0, keepdims=True) + 1e-6
	norm_labels = norm_labels.tolist()

	# Normalizing data:
	norm_data = np.asarray(data)
	norm_data -= np.mean(norm_data, axis=0, keepdims=True)
	norm_data /= np.std(norm_data, axis=0, keepdims=True) + 1e-6
	norm_data = norm_data.tolist()

	# Creating loader
	train_data = UCIDataSet(norm_labels, norm_data)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=2565, shuffle=False)

	return train_loader
