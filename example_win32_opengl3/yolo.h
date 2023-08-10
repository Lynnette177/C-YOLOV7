#pragma once
#include<iostream>
#include<opencv.hpp>


struct Output
{
    int id;             //结果类别id
    float confidence;   //结果置信度
    cv::Rect box;       //矩形框
};

class Yolo
{
public:
    Yolo() {}
    ~Yolo() {}
    bool readModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
    bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output);
    void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);
    unsigned long long int emotion[8] = { 0 };
    bool window = 1;
    int checkwindow = 0;
    int personnum = 0;
private:

    float sigmoid_x(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }


    const float netAnchors[3][6] = { {12, 16, 19, 36, 40, 28},{36, 75, 76, 55, 72, 146},{142, 110, 192, 243, 459, 401} }; //yolov7-P5 anchors

    const int netWidth = 640;   //ONNX图片输入宽度
    const int netHeight = 640;  //ONNX图片输入高度
    const int strideSize = 3;   //步长

    const float netStride[4] = { 8, 16.0,32,64 };

    float boxThreshold = 0.25;
    float classThreshold = 0.25;
    float nmsThreshold = 0.45;
    float nmsScoreThreshold = boxThreshold * classThreshold;

    std::vector<std::string> className = { "Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise" };
};
