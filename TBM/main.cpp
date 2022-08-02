// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <bits/stdc++.h>
#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
// #include <samples/classification_results.h>

#include <opencv2/opencv.hpp>
#include "info.h"

#ifndef BENCH_TXT
#include <mv_cam.h>
#endif

#define CLASS         3
#define BATCH_SIZE    1
#define IMAGE_WIDTH   224
#define IMAGE_HEIGHT  224
#define IMAGE_CHANNEL 3
#define IMAGE_SIZE    (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL)

using namespace InferenceEngine;


ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    std::cout << "Parsing input parameters" << '\n';
    return true;
}

template <typename T, typename U>
void matToBlob(const cv::Mat& cv_img, Blob::Ptr& blob) {
    SizeVector blobSize = blob.get()->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    T* blob_data = blob->buffer().as<T*>();

    cv::resize(cv_img, cv_img, cv::Size(width, height));
    // BGR...BGR...BGR --> RRR...GGG...BBB
    for (size_t c = 0; c < channels; c++)
        for (size_t  h = 0; h < height; h++)
            for (size_t w = 0; w < width; w++)
                blob_data[c * width * height + h * width + w] =
                        cv_img.at<U>(h, w)[c];
}

inline int readTestFile(std::ifstream &stream, const std::string test_txt) {
    stream.open(test_txt, std::ios::in | std::ios::binary);
    if (!stream.is_open()) return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    const char *core_plugin_dir = "/opt/intel/openvino_2019.1.094/deployment_tools/inference_engine/lib/intel64";
    // const char *extra_plugin_dir = "./lib";
    const char *plug_dev = "HETERO:FPGA,CPU";
    // const char *plug_dev = "CPU";
    const char *net_topo_slack, *bin_data_slack;

    if (strcmp(plug_dev, "HETERO:FPGA,CPU") == 0) {
        net_topo_slack = "/home/terasic/zjq/TBMData/IR/VGG11/FP16/vgg11.xml";
        bin_data_slack = "/home/terasic/zjq/TBMData/IR/VGG11/FP16/vgg11.bin";
    } else if (strcmp(plug_dev, "CPU") == 0) {
        net_topo_slack = "/home/terasic/zjq/TBMData/IR/VGG11/FP32/vgg11.xml";
        bin_data_slack = "/home/terasic/zjq/TBMData/IR/VGG11/FP32/vgg11.bin";
    }
#ifdef BENCH_TXT
    const char *test_txt_slack = "/home/terasic/zjq/TBMData/dataset/test_zjq.txt";
    double total = 0.0;
#endif
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << '\n';
        if (!ParseAndCheckCommandLine(argc, argv)) return 0;

        std::cout << "Loading plugin" << '\n';
        InferencePlugin plugin = PluginDispatcher({core_plugin_dir}).getPluginByDevice(plug_dev);
        InferencePlugin plugin_shit = PluginDispatcher({core_plugin_dir}).getPluginByDevice(plug_dev);
        
        if (FLAGS_p_msg) {
            static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
        }

        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            std::cout << "CPU Extension loaded: " << FLAGS_l << '\n';
        }

        /** Setting plugin parameter for collecting per layer metrics **/
        if (FLAGS_pc)
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        printPluginVersion(plugin, std::cout);
        
        CNNNetReader networkReaderSlack;
        networkReaderSlack.ReadNetwork(net_topo_slack);
        networkReaderSlack.ReadWeights(bin_data_slack);
        CNNNetwork network_slack = networkReaderSlack.getNetwork();
        
        // Configure input & output
        std::cout << "Preparing input blobs\n";

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfoSlack = network_slack.getInputsInfo();
        
        auto inputInfoItemSlack = *inputInfoSlack.begin();
        // inputInfoItemSlack.second->setPrecision(Precision::U8);
        inputInfoItemSlack.second->setPrecision(Precision::FP32);
        inputInfoItemSlack.second->setLayout(Layout::NCHW);
        
        network_slack.setBatchSize(BATCH_SIZE);

        std::cout << "Preparing output blobs\n";

        OutputsDataMap outputInfo(network_slack.getOutputsInfo());
        auto outputInfoItem = *outputInfo.begin();
        std::string firstOutputName = outputInfoItem.first;
        DataPtr outputData = outputInfoItem.second;
        outputInfoItem.second->setPrecision(Precision::FP32);

        if (!outputData) {
            throw std::logic_error("output data pointer is not valid\n");
        }

        const SizeVector outputDims = outputInfo.begin()->second->getDims();
        bool outputCorrect = false;
        if (outputDims.size() == 2 /* NC */) {
            outputCorrect = true;
        } else if (outputDims.size() == 4 /* NCHW */) {
            /* H = W = 1 */
            if (outputDims[2] == 1 && outputDims[3] == 1) outputCorrect = true;
        }
        if (!outputCorrect) {
            throw std::logic_error("Incorrect output dimensions for classification model");
        }

        std::cout << "Loading model to the plugin" << '\n';
        
        ExecutableNetwork executable_network_slack = plugin_shit.LoadNetwork(network_slack, {});

        // inputInfoItem.second = {};
        // outputInfo = {};
        // network = {};
        // networkReaderBearing = {};

        cv::Mat input_img_slack, result_slack;
        
        std::cout << "\nStarting inference ...";
        InferRequest infer_request_slack = executable_network_slack.CreateInferRequest();
    
#ifdef USE_ASYNC
        const int max_number_of_iterations = 8;
        int iterations = max_number_of_iterations;
        /** Set callback function for calling on completion of async request **/
        infer_request_slack.SetCompletionCallback(
                [&] {
                    std::cout << "\nCompleted " << max_number_of_iterations - iterations + 1
                                << " async request";
                    if (--iterations) {
                        /** Start async request (max_number_of_iterations - 1) more times **/
                        infer_request_slack.StartAsync();
                    }
                });
        infer_request_slack.StartAsync();
        /** Wait all repetition of async requests **/
        for (int i = 0; i < max_number_of_iterations; i++) {
            infer_request_slack.Wait(IInferRequest::WaitMode::RESULT_READY);
        }
#endif

        
#ifdef BENCH_TXT
        std::ifstream test_fstream;
        std::string name_slack;
        unsigned true_label, img_iteration = 0, predict_right = 0;
        
        if (!readTestFile(test_fstream, test_txt_slack))
            throw std::runtime_error("\ninit test slack dataset .txt failed\n");
        
        while (!test_fstream.eof()) {
            test_fstream >> name_slack >> true_label;
            // std::cout << name_slack << "\n";
            ++img_iteration;
            if (name_slack.empty()) break;
            // input cv::Mat order is BGR
            input_img_slack = cv::imread(name_slack, cv::IMREAD_COLOR);
#else // for MVCam
        MVCam cam;
        if (cam.init_mv_cam() != CAMERA_STATUS_SUCCESS) throw std::runtime_error("\ninit camera failed\n");
        
        while (!cam.shut_down) {
            cam.get();
            input_img_slack = cv::Mat(cam.sFrameHead.iHeight, cam.sFrameHead.iWidth, CV_8UC3, cam.g_pBgrBuffer);
            cv::resize(input_img_slack, input_img_slack, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
#endif

            result_slack = input_img_slack;
#ifdef SHOW_RESWINDOW
            cv::imshow("input image", input_img_slack);
            cv::moveWindow("input image", 400, 40);
            if (cv::waitKey(1) == 27) cam.shut_down = true;
#endif

            input_img_slack.convertTo(input_img_slack, CV_32FC3, 1. / 255);

            Blob::Ptr input_slack = infer_request_slack.GetBlob(inputInfoSlack.begin()->first);
            
            // matToBlob<PrecisionTrait<Precision::FP16>::value_type, cv::Vec3b>(input_img_slack, input);
            matToBlob<PrecisionTrait<Precision::FP32>::value_type, cv::Vec3f>(input_img_slack, input_slack);
#ifdef BENCH_TXT
            typedef std::chrono::high_resolution_clock Time;
            typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
            typedef std::chrono::duration<float> fsec;
            
            auto t0 = Time::now();
#endif
            infer_request_slack.Infer();
#ifdef BENCH_TXT
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
#endif
            // Process output
            const Blob::Ptr output_blob = infer_request_slack.GetBlob(firstOutputName);

            std::vector<unsigned> results;
            TopResults(CLASS, *output_blob, results);
#ifdef BENCH_TXT
            if (true_label == results[0]) predict_right++;
#endif
#ifdef SHOW_RESWINDOW
            std::string text = std::to_string(results[0]); //  + to_string(label);
            cv::putText(result_slack, "the result is: ", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(result_slack, text, cv::Point(30, 200), cv::FONT_HERSHEY_SIMPLEX, 6, cv::Scalar(0, 0, 255), 3);
            cv::imshow("result", result_slack);
            cv::moveWindow("result", 626, 40);
#endif

#ifndef BENCH_TXT // for MV cam
        }
        std::cout << "mission complete, now closing the camera...\n";
#else
        }
        std::cout << "mission complete, now closing the text file...\n";
                  << "\nTotal images: " << img_iteration
                  << "\nTrue predict result:" << predict_right
                  << "\tAcc is: " << (float) predict_right / img_iteration
                  << "\nTotal inference time: " << total << "ms"
                  << "\nThroughput: " << img_iteration * 1e3 / total << "FPS";
#endif
        goto EXIT;
    }

    catch (const std::exception& error) {
        slog::err << error.what() << '\n';
        return 1;
    }

    catch (...) {
        slog::err << "\nUnknown/internal exception happened.";
        return 1;
    }

    EXIT:
        std::cout << "\nExecution successful";
        return 0;
}
