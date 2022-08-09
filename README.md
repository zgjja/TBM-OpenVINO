# TBM-OpenVINO

## Project Introduction

Rock slag classification to predict the tool head wearing of the TBM (tunnel boring machine), and provide a reference for its replacement.

## Project Composition

The whole project contains the training program (Python3) and the final deployment program (C++). They are packed in two branches. For Python3 code, a virtual environment is highly advised and I also provide the requirement.txt file.

## Environment & Hardware

For the deployment program, I use HERO (heterogenous extensible robot platform) as my development machine, which contains a Intel i5-8700 CPU and a Intel Arria10 GX1150 FPGA. The program runs on Ubuntu16.04 with Intel OpenVINO Toolkit 2019 R1 (FPGA supported version).

The calculation (mainly MAC) is executed Arria10. But you can also deploy this on a Intel CPU or intergrated GPU, theoretically.

We use MV-SUA202GC-T cam as input image source, I also tried Kinect before but it is not included in this project, if you only want to benchmark this, you can change the macro in the CMakeLists.txt

## Network Design & Simple Benchmark

In my own HERO platform, I use VGG11 as my base CNN, and I only use one fully connected layer as classifier for performance consideration. After training, the network are transformed to ONNX format first, then use the tool in OpenVINO to change it to the IR format that OpenVINO can recognize. In this two transformations, FP16 quantization and layer merge optimization are performed.

|Device|Est. Power|FPS|Throughput(GOPS)|Power Effi.(GOPS/W)|Performance|
|:----:|:----:|:----:|:----:|:----:|:----:|
|i5-8700|65W|10|149.70|2.303|X1|
|GTX-1080|180W|170|$\color{red}2544.90$|14.138|X6.13|
|Arria10GX|30W|153.5|2286.51|$\color{red}76.217$|33.09|
<!-- The final performance reaches 153.5 FPS, which is close to the NVIDIA GTX1080 implementation (180 FPS). Power consumption for Arria10 is lower than 30 Watts and the whole HERO platform is lower than 70 Watts, which is lower than the 180 Watts NVIDIA GTX1080 counterpart -->