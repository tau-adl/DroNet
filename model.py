import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Sampler
from sklearn import svm, metrics
from skimage import io, transform
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import json
import pandas
import csv
import datetime
import matplotlib.pyplot as plt


LEARNING_RATE = 0.005
NUMBER_OF_EPOCHS = 1
WEIGHT_DECAY_RATE = 0.01
DATA_BATCH_SIZE = 10
DROPOUT_PROBABILITY = 0.5

INPUT_CHANNELS = 3

CHANNEL_FACTOR = 0.25

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOGS_DIR = "./Logs/" + str(datetime.datetime.now().isoformat())
TRAIN_DATA_DIR = "Train/"
TEST_DATA_DIR = "Test/"


class DroNet(nn.Module):
  '''
  DroNet Netwrok
  '''
  def __init__(self):
    super().__init__()

    # Layer 0
    self.layer_0_conv1 = nn.Conv2d(INPUT_CHANNELS, int(32*CHANNEL_FACTOR), 5, stride=2, padding=2)
    self.layer_0_maxpool1 = nn.MaxPool2d(3, stride=2)

    # Layer 1
    self.layer_1_1_conv1 = nn.Conv2d(int(32*CHANNEL_FACTOR), int(32*CHANNEL_FACTOR), 3, stride=2, padding=1)
    self.layer_1_1_conv2 = nn.Conv2d(int(32*CHANNEL_FACTOR), int(32*CHANNEL_FACTOR), 3, padding=1)
    self.layer_1_2_conv1 = nn.Conv2d(int(32*CHANNEL_FACTOR), int(32*CHANNEL_FACTOR), 1, stride=2)

    # Layer 2
    self.layer_2_1_conv1 = nn.Conv2d(int(32*CHANNEL_FACTOR), int(64*CHANNEL_FACTOR), 3, stride=2, padding=1)
    self.layer_2_1_conv2 = nn.Conv2d(int(64*CHANNEL_FACTOR), int(64*CHANNEL_FACTOR), 3, padding=1)
    self.layer_2_2_conv1 = nn.Conv2d(int(32*CHANNEL_FACTOR), int(64*CHANNEL_FACTOR), 1, stride=2)

    # Layer 3
    self.layer_3_1_conv1 = nn.Conv2d(int(64*CHANNEL_FACTOR), int(128*CHANNEL_FACTOR), 3, stride=2, padding=1)
    self.layer_3_1_conv2 = nn.Conv2d(int(128*CHANNEL_FACTOR), int(128*CHANNEL_FACTOR), 3, padding=1)
    self.layer_3_2_conv1 = nn.Conv2d(int(64*CHANNEL_FACTOR), int(128*CHANNEL_FACTOR), 1, stride=2)

    # Layer 4
    self.layer_4_dropout = nn.Dropout(DROPOUT_PROBABILITY)
    self.layer_4_linear = nn.Linear(7 * 10 * int(128*CHANNEL_FACTOR), int(256*CHANNEL_FACTOR))

    # Layer 5
    self.layer_5_linear = nn.Linear(int(256*CHANNEL_FACTOR), 3)


  def forward(self, x):
    # Layer 0
    x0 = self.layer_0_conv1(x)
    x0 = self.layer_0_maxpool1(x0)

    ##########
    # Layer 1
    ##########
    # Layer 1_1
    x11 = F.relu(x0)
    x11 = self.layer_1_1_conv1(x11)
    x11 = F.relu(x11)
    x11 = self.layer_1_1_conv2(x11)

    # Layer 1_2
    x12 = self.layer_1_2_conv1(x0)

    # Layer 1 Total
    x11.add(x12)
    x1 = x11

    ##########
    # Layer 2
    ##########
    # Layer 2_1
    x21 = F.relu(x1)
    x21 = self.layer_2_1_conv1(x21)
    x21 = F.relu(x21)
    x21 = self.layer_2_1_conv2(x21)

    # Layer 2_2
    x22 = self.layer_2_2_conv1(x1)

    # Layer 2 Total
    x21.add(x22)
    x2 = x21

    ##########
    # Layer 3
    ##########
    # Layer 3_1
    x31 = F.relu(x2)
    x31 = self.layer_3_1_conv1(x31)
    x31 = F.relu(x31)
    x31 = self.layer_3_1_conv2(x31)

    # Layer 2_2
    x32 = self.layer_3_2_conv1(x2)

    # Layer 2 Total
    x31.add(x32)
    x3 = x31

    ##########
    # Layer 4
    ##########
    x4 = torch.reshape(x3, (DATA_BATCH_SIZE, -1,))
    x4 = self.layer_4_dropout(x4)
    x4 = self.layer_4_linear(x4)

    ##########
    # Layer 5
    ##########
    x5 = self.layer_5_linear(F.relu(x4))

    return x5


# def save_results(results, file_path):
#   pandas.DataFrame.from_dict(
#         results, 
#         orient = 'columns',
#     ).to_csv(f'{file_path}.csv')

#   with open(f'{file_path}.json', 'w', encoding='utf-8') as fd:
#     json.dump(results, fd, ensure_ascii=False, indent=4)


def plot_convergence_graph(results, legend=None):
    if not legend:
        legend = range(len(results))
    
    losses = []
    for i, result in enumerate(results):
      losses.append([])
      losses[i].append([])
      losses[i].append([])
      for epoch_result in result:
        losses[i][0].append(epoch_result["train loss"])
        losses[i][1].append(epoch_result["test loss"])

    for i, losses in enumerate(losses):
        train_loss, test_loss = losses

        plt.figure(i)
        plt.plot(range(1, NUMBER_OF_EPOCHS+1),train_loss)
        plt.plot(range(1, NUMBER_OF_EPOCHS+1),test_loss, color='r')
        plt.title(legend[i] + " Loss (blue-train, red-test)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid()

        plt.show()
 

def show_image(images, classes):
    np_images = images.numpy()
    for i, np_img in enumerate(np_images):
        plt.subplot(1, len(np_images), i+1)
        plt.imshow(np_img[0], cmap='gray')
        plt.title(CLASSES[classes[i]])

    plt.show()


def train_net(net, train_loader, test_loader, weight_decay=0, tensor_board_path=None):
    # Defining loss function:
    criterion = nn.MSELoss()

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


class DroneImagesDataSet(torch.utils.data.Dataset):
    """Drone Images dataset"""

    def __init__(self, labels_path, root_dir, transform=None):
        """
        Args:
            labels_path (string): Path to the labels file.
            root_dir (string): Directory with all the images.
        """
        self.labels_path = labels_path
        self.root_dir = root_dir
        self.transform = transform

        with open(labels_path, 'r') as fd:
        	labels = fd.read()
        	labels = labels.split()

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx].split(";")
        img_name = os.path.join(self.root_dir, label[0]+".jpg")
        image = io.imread(img_name)
        image = np.rollaxis(image, 2, 0)
        label = np.array(label[1::]).astype('float')
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def train_and_evaluate():
	train_data = DroneImagesDataSet(labels_path=TRAIN_DATA_DIR + 'labels.txt', root_dir=TRAIN_DATA_DIR)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=DATA_BATCH_SIZE, shuffle=False)

	test_data = DroneImagesDataSet(labels_path=TEST_DATA_DIR + 'labels.txt', root_dir=TEST_DATA_DIR)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=DATA_BATCH_SIZE, shuffle=False)

	net = DroNet()
	net.to(DEVICE)
	results = train_net(net, train_loader, test_loader, tensor_board_path=LOGS_DIR)

	# Saving our training model:
	path = os.path.join(LOGS_DIR, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")))
	torch.save(net.state_dict(), path)
	# save_results(results, path)