import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 3
CHANNEL_FACTOR = 0.25
DROPOUT_PROBABILITY = 0.5
DATA_BATCH_SIZE = 10
RESIDUAL_WIDTH = 3


class MultiResDroNet(nn.Module):
  '''
  DroNet Netwrok with multi residual blocks.
  '''
  def __init__(self, input_channels=INPUT_CHANNELS, channel_factor=CHANNEL_FACTOR, dropout_probability=DROPOUT_PROBABILITY, batch_size=DATA_BATCH_SIZE):
    super().__init__()

    # Layer 0
    self.layer_0_conv1 = nn.Conv2d(input_channels, int(32*channel_factor), 5, stride=2, padding=2)
    self.layer_0_maxpool1 = nn.MaxPool2d(3, stride=2)

    # Layer 1
    self.layer_1_res_blocks = nn.ModuleList([])
    for res_block_idx in range(RESIDUAL_WIDTH):
      self.layer_1_res_blocks.append(nn.ModuleList([]))

      self.layer_1_res_blocks[res_block_idx].append(nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, stride=2, padding=1))
      self.layer_1_res_blocks[res_block_idx].append(nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, padding=1))
      self.layer_1_res_blocks[res_block_idx].append(nn.BatchNorm2d(int(32*channel_factor)))

    self.layer_1_skip_conv = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 1, stride=2)

    # Layer 2
    self.layer_2_res_blocks = nn.ModuleList([])
    for res_block_idx in range(RESIDUAL_WIDTH):
      self.layer_2_res_blocks.append(nn.ModuleList([]))

      self.layer_2_res_blocks[res_block_idx].append(nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 3, stride=2, padding=1))
      self.layer_2_res_blocks[res_block_idx].append(nn.Conv2d(int(64*channel_factor), int(64*channel_factor), 3, padding=1))
      self.layer_2_res_blocks[res_block_idx].append(nn.BatchNorm2d(int(64*channel_factor)))

    self.layer_2_skip_conv = nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 1, stride=2)

    # Layer 3
    self.layer_3_res_blocks = nn.ModuleList([])
    for res_block_idx in range(RESIDUAL_WIDTH):
      self.layer_3_res_blocks.append(nn.ModuleList([]))

      self.layer_3_res_blocks[res_block_idx].append(nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 3, stride=2, padding=1))
      self.layer_3_res_blocks[res_block_idx].append(nn.Conv2d(int(128*channel_factor), int(128*channel_factor), 3, padding=1))
      self.layer_3_res_blocks[res_block_idx].append(nn.BatchNorm2d(int(128*channel_factor)))

    self.layer_3_skip_conv = nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 1, stride=2)

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
    x1 = self.layer_1_skip_conv(x0)
    for res_block in self.layer_1_res_blocks:
      x11 = F.relu(x0)
      x11 = res_block[0](x11)
      x11 = F.relu(x11)
      x11 = res_block[1](x11)
      x11 = res_block[2](x11)
      x1.add(x11)

    ##########
    # Layer 2
    ##########
    x2 = self.layer_2_skip_conv(x1)
    for res_block in self.layer_2_res_blocks:
      x21 = F.relu(x1)
      x21 = res_block[0](x21)
      x21 = F.relu(x21)
      x21 = res_block[1](x21)
      x21 = res_block[2](x21)
      x2.add(x21)

    ##########
    # Layer 3
    ##########
    x3 = self.layer_3_skip_conv(x2)
    for res_block in self.layer_3_res_blocks:
      x31 = F.relu(x2)
      x31 = res_block[0](x31)
      x31 = F.relu(x31)
      x31 = res_block[1](x31)
      x31 = res_block[2](x31)
      x3.add(x31)

    ##########
    # Layer 4
    ##########
    x4 = torch.reshape(x3, (x3.shape[0], -1,))
    x4 = self.layer_4_dropout(x4)
    x4 = self.layer_4_linear(x4)

    ##########
    # Layer 5
    ##########
    x5 = self.layer_5_linear(F.relu(x4))

    return x5