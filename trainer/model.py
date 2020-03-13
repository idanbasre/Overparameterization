from . import LinearNDepthNet
from . import UCIDataSet
from . import Utils

import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

import os
import datetime


ETHANOL_LABEL = "1"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_net(net, train_loader, num_epochs, P, lr):
    # Defining optimizer:
    optimizer = optim.SGD(net.parameters(), lr=lr)  

    # Start Training
    results = []

    for epoch in range(num_epochs):
      epoch_start_time = datetime.datetime.now()
      net = net.train()
      train_loss = 0
      for i, data in enumerate(train_loader, 0):
        x, y = data[0].to(DEVICE), data[1].to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        y_pred = net(x)
        y_pred = y_pred.view(-1)

        # loss
        loss = (1/P)*(y_pred-y)**P
        avg_loss = loss.mean()

        # backward
        avg_loss.backward()
        train_loss += avg_loss
        optimizer.step()

      epoch_end_time = datetime.datetime.now()
      # Result dict for saving data
      epoch_result = {}
      epoch_result["epoch"] = epoch
      epoch_result["train loss"] = train_loss.item()
      epoch_result["duration"] = str(epoch_end_time - epoch_start_time)
      results.append(epoch_result)

      if epoch % 1000 == 0:
        print("Finished epoch number: {}, loss is: {}".format(epoch, train_loss))

    return results


def train_and_evaluate(train_data_dir, log_dir, layer_width, P, num_epochs, lr):
    # Preparing Logs dir:
    description = Utils.create_architecture_description(layer_width, P, num_epochs, lr)
    logs_dir = os.path.join(log_dir, description + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    os.mkdir(logs_dir)
    print('Logs saved at: {}'.format(logs_dir))

    # Loading Data and Train:
    train_loader = UCIDataSet.get_UCI_data_loader(train_data_dir, ETHANOL_LABEL)

    net = LinearNDepthNet.linear_n_dpeth(layer_width)
    net.to(DEVICE)

    results = train_net(net, train_loader, num_epochs, P, lr)

    # Saving the training model:
    path = os.path.join(logs_dir, description)
    torch.save(net.state_dict(), path)
    Utils.save_results(results, path)
