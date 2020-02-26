# DroNet

This repo implements DroNet network as described in:
* [Deep Drone Racing: from Simulation to Reality with Domain Randomization / Antonio Loquercio, Elia Kaufmann, Rene Ranftl, Alexey Dosovitskiy, Vladlen Koltun, and Davide Scaramuzza](http://rpg.ifi.uzh.ch/docs/TRO19_Loquercio.pdf).
* [DroNet: Learning to Fly by Driving / Antonio Loquercio, Ana I. Maqueda, Carlos R. del-Blanco, and Davide Scaramuzza](http://rpg.ifi.uzh.ch/docs/RAL18_Loquercio.pdf)

In addition, this repo was created as part of Deep Learning course in Tel-Aviv University (Course Number: 0510-7255).

## Implemented Networks:

### DroNet
This network is the ruglar DroNet (implemented in Pytorch).
It is describe at DroNet: Learning to Fly by Driving / Antonio Loquercio, Ana I. Maqueda, Carlos R. del-Blanco, and Davide Scaramuzza.

### MultiRes DroNet
This network takes DroNet and adds Residaul Blocks in parallel.
You can change the number of addidional Residual Blocks to add in the file "MultiResDroNet.py" (RESIDUAL_WIDTH parameter).

### Deep DroNet
This network takes DroNet and adds Residual Blocks after the 3 Residual Blocks that are already in the regular DroNet.
You can change the number of additional Residual Blocks to add in the file "DeepDroNet.py" ("ADDITIONAL_LAYERS parameter).

## Data
You can use the offline data that was recorded from UZH Robotics and Perception Group (in their simulation that was implemnted in the repo https://github.com/uzh-rpg/sim2real_drone_racing).
The data can be found at: http://rpg.ifi.uzh.ch/datasets/sim2real_ddr/simulation_training_data.zip.

In addition, you can generate new simulated data (instruction in the sim2real_drone_racing repo). 

## Training Network
1. Open a directory for Logs.
2. In run.sh file (before specify in the file the locations for the followig directories):
- TRAIN_DATA_PATH
- TEST_DATA_PATH
- OUTPUT_DIR (i.e. Logs dir)
