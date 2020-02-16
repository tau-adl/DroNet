from . import DroNet
from . import DataSet
from . import Utils

import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

import os
import datetime


NUMBER_OF_EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY_RATE = 0.01
DATA_BATCH_SIZE = 10
DROPOUT_PROBABILITY = 0.5

INPUT_CHANNELS = 3
CHANNEL_FACTOR = 0.25

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOGS_DIR = "Logs"
TRAIN_DATA_DIR = os.path.join("Small_Data", "Train")
TEST_DATA_DIR = os.path.join("Small_Data", "Test")


def train_net(net, train_loader, test_loader, weight_decay=0, tensor_board_path=None):
    # Defining loss function:
    criterion = torch.nn.MSELoss()

    # Defining optimizer:
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    # Start Training
    results = []
    # writer = SummaryWriter(tensor_board_path)

    for epoch in range(NUMBER_OF_EPOCHS):
      epoch_start_time = datetime.datetime.now()
      net = net.train()
      for i, data in enumerate(train_loader, 0):   
        inputs, labels = data["image"].to(DEVICE, dtype=torch.float), data["label"].to(DEVICE, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

      # calculating  accuracy
      loss_train = get_loss(net, train_loader)
      loss_test = get_loss(net, test_loader)

      epoch_end_time = datetime.datetime.now()
      # Result dict for saving data
      epoch_result = {}
      epoch_result["epoch"] = epoch
      epoch_result["train loss"] = loss_train.item()
      epoch_result["test loss"] = loss_test.item()
      epoch_result["duration"] = str(epoch_end_time - epoch_start_time)
      results.append(epoch_result)

      # Record to TensorBoard
      # writer.add_scalar('Train Loss', loss_train, epoch)
      # writer.add_scalar('Test Loss', loss_test, epoch)

      # for name, param in net.named_parameters():
        # writer.add_histogram(name, param, epoch)

    # writer.close()
    return results


def get_loss(net, data_loader):
    loss = 0
    net = net.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data["image"].to(DEVICE, dtype=torch.float), data["label"].to(DEVICE, dtype=torch.float)
            outputs = net(inputs)
            
            loss += torch.nn.functional.mse_loss(outputs, labels)

    return loss / (i + 1)


def train_and_evaluate():
    os.mkdir(LOGS_DIR)
    train_labels_path = os.path.join(TRAIN_DATA_DIR, 'labels.txt')
    train_data = DataSet.DroneImagesDataSet(labels_path=train_labels_path, root_dir=TRAIN_DATA_DIR)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=DATA_BATCH_SIZE, shuffle=False)

    test_labels_path = os.path.join(TEST_DATA_DIR, 'labels.txt')
    test_data = DataSet.DroneImagesDataSet(labels_path=test_labels_path, root_dir=TEST_DATA_DIR)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=DATA_BATCH_SIZE, shuffle=False)

    net = DroNet.DroNet(input_channels=INPUT_CHANNELS, channel_factor=CHANNEL_FACTOR, dropout_probability=DROPOUT_PROBABILITY, batch_size=DATA_BATCH_SIZE)
    net.to(DEVICE)

    print("Start training at: " + str(datetime.datetime.now().isoformat()))
    results = train_net(net, train_loader, test_loader, tensor_board_path=LOGS_DIR)
    print("Finish training at: " + str(datetime.datetime.now().isoformat()))

    # Saving our training model:
    path = os.path.join(LOGS_DIR, "net")
    torch.save(net.state_dict(), path)
    Utils.save_results(results, path)
