from . import LinearNDepthNet
from . import UCIDataSet
from . import Utils

import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

import os
import datetime


IN_DIM = 128
OUT_DIM = 1
LAYERS_WIDTH = [IN_DIM, 10, 10, OUT_DIM]
N_LAYERS = len(LAYERS_WIDTH) - 1
P = 2 # Order of loss function (l_p loss)
NUMBER_OF_EPOCHS = 10
LEARNING_RATE = 0.001
ETHANOL_LABEL = "1"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_net(net, train_loader, tensor_board_path=None):
    criterion = torch.nn.MSELoss()
    # Defining optimizer:
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)  

    # Start Training
    results = []
    # writer = SummaryWriter(tensor_board_path)

    for epoch in range(NUMBER_OF_EPOCHS):
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

      # Record to TensorBoard
      # writer.add_scalar('Train Loss', train_loss, epoch)

      # for name, param in net.named_parameters():
        # writer.add_histogram(name, param, epoch)

    # writer.close()
    return results


def train_and_evaluate(train_data_dir, log_dir):
    # Preparing Logs dir:
    description = Utils.create_architecture_description(LAYERS_WIDTH, P, NUMBER_OF_EPOCHS, LEARNING_RATE)
    logs_dir = os.path.join(log_dir, description + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    os.mkdir(logs_dir)

    # Loading Data and Train:
    train_loader = UCIDataSet.get_UCI_data_loader(train_data_dir, ETHANOL_LABEL)

    net = LinearNDepthNet.linear_n_dpeth(LAYERS_WIDTH)
    net.to(DEVICE)

    results = train_net(net, train_loader, tensor_board_path=logs_dir)

    # Saving the training model:
    path = os.path.join(logs_dir, "net")
    torch.save(net.state_dict(), path)
    Utils.save_results(results, path)
