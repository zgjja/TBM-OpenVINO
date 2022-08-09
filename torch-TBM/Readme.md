# Subproject -- Training Scripts in Python3

## Functionalities

* Train and test the dataset, the training and testing part are included in train.py;
* in my project, i use VGG-11 and VGG-16 as my network, and i only reserve one fc layer at the end of the network for the consideration of FPGA deployment;
* dataloader, ONNX converter are also included in this Python3 subproject.

## Some Features and Usage

### train.py

* Support FP16 mixed precision training;
* You can develope your own network and plug it in;
* You can run this script to start training or testing;
* Opptimizer, lr scheduler can be replaced to make further test.

### dataloader.py

* You can use the transform method PyTorch provides or write your own;
* For different datasets, you need to read and modify the dataloader file (dataloader.py);
* Some more work maybe need to be done for your own datasets, such as calculating mean and std, etc.

### convertToONNX.py

* Plug in your own network and make a full inference process, PyTorch will expoet your net with ONNX format;
* You can export FP32/FP16 or even INT8(not implemented in this project)