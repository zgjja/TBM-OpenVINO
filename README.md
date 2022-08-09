# TBM-OpenVINO

## Project Introduction

Rock slag classification to predict the tool head wearing of the TBM (tunnel boring machine), and provide a reference for its replacement.

## Project Composition

The whole project contains the training program (Python3) and the final deployment program (C++). They are packed in two branches. For Python3 code, a virtual environment is highly advised and I also provide the requirement.txt file.

## Environment & Hardware

* OS
    * Ubuntu16.04

* Deployment Env. & Hardware
    * Intel OpenVINO toolkit 2019R1
    * Intel FPGA SDK for OpenCL
    * .AOCX file (precompiled OpenCL bitstream, provided by platform vendor)
    * gcc for Ubuntu16.04, CMake
    * HERO (heterogenous extensible robot platform)
        * Arria10 GX1150 FPGA
        * MV-SUA202GC-T (Kinect is also tested but not included in this repo)

* Training Env. & Hardware
    * Python3.5
    * PyTorch1.12.0+cu116
    * i5-8700 + GTX1080

## Network Design & Simple Benchmark

In my own HERO platform, I use VGG11 as my base CNN, and I only use one fully connected layer as classifier for performance consideration. After training, the network are transformed to ONNX format first, then use the tool in OpenVINO to change it to the IR format that OpenVINO can recognize. In this two transformations, FP16 quantization and layer merge optimization are performed.

|Device|Est. Power(W)|FPS|Throughput(GOPS)|Power Effi.(GOPS/W)|Performance|
|:----:|:----:|:----:|:----:|:----:|:----:|
|i5-8700|65|10|149.70|2.303|X1|
|GTX-1080|180|170|$\color{red}2544.90$|14.138|X6.13|
|Arria10 GX1150|30|153.5|2286.51|$\color{red}76.217$|X33.09|

* The overall HERO platform power consumption is 70 Watts, so the overall power efficiency is about 32.68GOPSGOPS/W.