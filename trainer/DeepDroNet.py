import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 3
CHANNEL_FACTOR = 0.25
DROPOUT_PROBABILITY = 0.5
DATA_BATCH_SIZE = 10
ADDITIONAL_LAYERS = 4


class DeepDroNet(nn.Module):
  '''
  DroNet Netwrok with more layers.
  '''
  def __init__(self, input_channels=INPUT_CHANNELS, channel_factor=CHANNEL_FACTOR, dropout_probability=DROPOUT_PROBABILITY, batch_size=DATA_BATCH_SIZE):
    super().__init__()

    # Layer 0
    self.layer_0_conv1 = nn.Conv2d(input_channels, int(32*channel_factor), 5, stride=2, padding=2)
    self.layer_0_maxpool1 = nn.MaxPool2d(3, stride=2)

    # Layer 1
    self.layer_1_1_conv1 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, stride=2, padding=1)
    self.layer_1_1_conv2 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, padding=1)
    self.bn_1 = nn.BatchNorm2d(int(32*channel_factor))
    self.layer_1_2_conv1 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 1, stride=2)

    # Layer 2
    self.layer_2_1_conv1 = nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 3, stride=2, padding=1)
    self.layer_2_1_conv2 = nn.Conv2d(int(64*channel_factor), int(64*channel_factor), 3, padding=1)
    self.bn_2 = nn.BatchNorm2d(int(64*channel_factor))
    self.layer_2_2_conv1 = nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 1, stride=2)

    # Layer 3
    self.layer_3_1_conv1 = nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 3, stride=2, padding=1)
    self.layer_3_1_conv2 = nn.Conv2d(int(128*channel_factor), int(128*channel_factor), 3, padding=1)
    self.bn_3 = nn.BatchNorm2d(int(128*channel_factor))
    self.layer_3_2_conv1 = nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 1, stride=2)

    # Additional layers
    self.additional_layers = nn.ModuleList([])
    for layer_idx in range(ADDITIONAL_LAYERS):
      self.additional_layers.append(nn.ModuleList([]))

      self.additional_layers[layer_idx].append(nn.Conv2d(int(128*channel_factor), int(128*channel_factor), 1)) # skip connection

      self.additional_layers[layer_idx].append(nn.Conv2d(int(128*channel_factor), int(128*channel_factor), 3, padding=1))
      self.additional_layers[layer_idx].append(nn.Conv2d(int(128*channel_factor), int(128*channel_factor), 3, padding=1))
      self.additional_layers[layer_idx].append(nn.BatchNorm2d(int(128*channel_factor)))

    # Layer 4
    self.layer_4_dropout = nn.Dropout(dropout_probability)
    self.layer_4_linear = nn.Linear(7 * 10 * int(128*channel_factor), int(256*channel_factor))

    # Layer 5
    self.layer_5_linear = nn.Linear(int(256*channel_factor), 3)


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
    x11 = self.bn_1(x11)

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
    x21 = self.bn_2(x21)

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
    x31 = self.bn_3(x31)

    # Layer 2_2
    x32 = self.layer_3_2_conv1(x2)

    # Layer 2 Total
    x31.add(x32)
    x3 = x31

    ###################
    # Additional Layers
    ###################
    x_additional = x3
    for (i, layer) in enumerate(self.additional_layers):
      x_skip = layer[0](x_additional) # skip connection
      x41 = F.relu(x_additional)
      x41 = layer[1](x41)
      x41 = F.relu(x41)
      x41 = layer[2](x41)
      x41 = layer[3](x41)
      x_additional = x_skip
      x_additional.add(x41)
    
    ##########
    # Layer 4
    ##########
    x4 = torch.reshape(x_additional, (x_additional.shape[0], -1,))
    x4 = self.layer_4_dropout(x4)
    x4 = self.layer_4_linear(x4)

    ##########
    # Layer 5
    ##########
    x5 = self.layer_5_linear(F.relu(x4))

    return x5
