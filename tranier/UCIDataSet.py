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


def get_data_from_file(file_path, desired_label):
  with open(file_path) as fd:
    samples = fd.read()
    samples = samples.split("\n")

  data = []
  labels = []
  for sample in samples[:-1]:
    label, vector = sample.split(";")
    if label is not desired_label:
      continue

    label = float(label)
    labels.append(label)

    vector = vector.split(" ")[1:-1]
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
	norm_labels = [l - 1.0 for l in labels]

	# Normalizing data:
	norm_data = np.asarray(data)
	norm_data -= np.mean(norm_data, axis=0, keepdims=True)
	norm_data /= np.std(norm_data, axis=0, keepdims=True) + 1e-6
	norm_data = norm_data.tolist()

	# Creating loader
	train_data = UCIDataSet(norm_labels, norm_data)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=2565, shuffle=False)

	return train_loader
