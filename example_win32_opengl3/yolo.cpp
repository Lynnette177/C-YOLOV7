#include"yolo.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

bool Yolo::readModel(Net& net, string& netPath, bool isCuda = true)
{
    try
    {
        net = readNet(netPath);
    }
    catch (const std::exception&)
    {
        return false;
    }
    //cuda
    if (isCuda)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    //cpu
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return true;
}

bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& output)
{
    Mat blob;
    int col = SrcImg.cols;
    int row = SrcImg.rows;
    int maxLen = MAX(col, row);
    Mat netInputImg = SrcImg.clone();
    if (maxLen > 1.2 * col || maxLen > 1.2 * row)
    {
        Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
        SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
        netInputImg = resizeImg;
    }
    vector<Ptr<Layer> > layer;
    vector<string> layer_names;
    layer_names = net.getLayerNames();
    blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
    /*让我们来解释一下参数：
    netInputImg：这是你想要进行预处理的输入图像。它应该是cv::Mat类型，这是OpenCV中表示图像的矩阵数据结构。
    blob：这是用于存储预处理图像的输出blob。Blob是一个多维数组，常用于深度学习框架中高效地存储和处理大量数据。
    1 / 255.0：这个参数是缩放因子，用于对图像的像素值进行归一化。除以255.0将像素值缩放到0到1的范围内，这是神经网络常用的归一化技术。
    cv::Size(netWidth, netHeight)：这个参数指定了输入图像将被调整大小的空间尺寸。netWidth和netHeight表示神经网络期望的输入图像的宽度和高度。
    cv::Scalar(0, 0, 0)：这是从图像的每个通道中减去的均值。在这种情况下，(0, 0, 0)表示从每个通道中减去零，从而将图像居中。
    true：这个参数表示是否交换输入图像的颜色通道。OpenCV以BGR（蓝绿红）格式读取图像，而许多深度学习模型期望图像以RGB（红绿蓝）格式。将这个参数设置为true会相应地交换通道。
    false：这个参数表示是否裁剪调整大小的图像的中心区域，以确保精确匹配所需的空间尺寸。将其设置为false表示将使用整个调整大小的图像，不进行裁剪。
    这个函数调用将对输入图像进行归一化、调整大小、均值减法和通道交换，并将结果存储在一个可以用作神经网络输入的blob中。*/
    net.setInput(blob);
    std::vector<cv::Mat> netOutputImg;
    net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
    std::vector<int> classIds;//结果id数组
    std::vector<float> confidences;//结果每个id对应置信度数组
    std::vector<cv::Rect> boxes;//每个id矩形框
    float ratio_h = (float)netInputImg.rows / netHeight;
    float ratio_w = (float)netInputImg.cols / netWidth;
    int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
    for (int stride = 0; stride < strideSize; stride++) {    //stride
        float* pdata = (float*)netOutputImg[stride].data;
        int grid_x = (int)(netWidth / netStride[stride]);
        int grid_y = (int)(netHeight / netStride[stride]);
        for (int anchor = 0; anchor < 3; anchor++) {	//anchors
            const float anchor_w = netAnchors[stride][anchor * 2];
            const float anchor_h = netAnchors[stride][anchor * 2 + 1];
            for (int i = 0; i < grid_y; i++) {
                for (int j = 0; j < grid_x; j++) {
                    float box_score = sigmoid_x(pdata[4]); ;//获取每一行的box框中含有某个物体的概率
                    if (box_score >= boxThreshold) {
                        cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
                        Point classIdPoint;
                        double max_class_socre;
                        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                        max_class_socre = sigmoid_x(max_class_socre);
                        if (max_class_socre >= classThreshold) {
                            float x = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
                            float y = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
                            float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
                            float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
                            int left = (int)(x - 0.5 * w) * ratio_w + 0.5;
                            int top = (int)(y - 0.5 * h) * ratio_h + 0.5;
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back(max_class_socre * box_score);
                            boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
                        }
                    }
                    pdata += net_width;//下一行
                }
            }
        }
    }
    /*这段代码是一个目标检测算法的主要循环，用于从神经网络的输出中提取检测到的目标框和相关信息。
for (int stride = 0; stride < strideSize; stride++)：对于每个stride（步长），循环迭代。
float* pdata = (float*)netOutputImg[stride].data;：获取当前stride下神经网络输出的数据指针。
int grid_x = (int)(netWidth / netStride[stride]);：计算当前stride下x方向的网格数量。
int grid_y = (int)(netHeight / netStride[stride]);：计算当前stride下y方向的网格数量。
for (int anchor = 0; anchor < 3; anchor++)：对于每个anchor（锚框），循环迭代。
const float anchor_w = netAnchors[stride][anchor * 2];：获取当前stride和anchor的宽度。
const float anchor_h = netAnchors[stride][anchor * 2 + 1];：获取当前stride和anchor的高度。
for (int i = 0; i < grid_y; i++)：对于每个y方向上的网格，循环迭代。
for (int j = 0; j < grid_x; j++)：对于每个x方向上的网格，循环迭代。
float box_score = sigmoid_x(pdata[4]);：获取当前网格的目标框得分（box_score），使用sigmoid函数对得分进行处理。
if (box_score >= boxThreshold)：如果目标框得分大于等于设定的阈值（boxThreshold），执行以下操作。
cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);：创建一个cv::Mat对象，用于存储当前网格的类别得分。
Point classIdPoint;：定义一个Point对象，用于存储最大类别得分的坐标信息。
double max_class_score;：定义一个变量，用于存储最大类别得分的值。
minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);：找到类别得分中的最大值和其坐标。
max_class_score = sigmoid_x(max_class_score);：对最大类别得分进行sigmoid函数处理。
if (max_class_score >= classThreshold)：如果最大类别得分大于等于设定的阈值（classThreshold），执行以下操作。
float x = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];：计算目标框的x坐标。
float y = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];：计算目标框的y坐标。
float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;：计算目标框的宽度。
float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;：计算目标框的高度。
int left = (int)(x - 0.5 * w) * ratio_w + 0.5;：计算目标框的左边界。
int top = (int)(y - 0.5 * h) * ratio_h + 0.5;：计算目标框的上边界。
classIds.push_back(classIdPoint.x);：将最大类别得分的类别标识（classIdPoint.x）添加到classIds容器中。
confidences.push_back(max_class_score * box_score);：将最大类别得分和目标框得分的乘积添加到confidences容器中。
boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));：将构建的目标框矩形添加到boxes容器中。
pdata += net_width;：将数据指针移动到下一行。
根据预测结果和设定的阈值，提取检测到的目标框和相关信息，并将它们存储在相应的容器中，以便后续的处理和显示。*/
    //执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
    /*用于执行非最大抑制（Non-Maximum Suppression）操作。让我们解释一下代码的功能：
boxes：这是一个包含所有目标框的矩形列表。
confidences：这是与每个目标框相关的置信度（confidence）列表。
nmsScoreThreshold：这是一个阈值，用于筛选掉置信度低于该阈值的目标框。
nmsThreshold：这是非最大抑制的重叠阈值。它用于确定两个目标框被视为重叠的程度。
nms_result：这是输出的结果列表，用于存储通过非最大抑制操作选中的目标框的索引。
该函数的作用是对boxes中的目标框进行非最大抑制处理，去除重叠度较高且置信度较低的目标框，保留置信度较高且重叠度较低的目标框。
执行完这行代码后，nms_result列表将包含通过非最大抑制操作选中的目标框的索引，这些索引可以用于获取选中的目标框和相关信息。*/
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Output result;
        result.id = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    if (output.size())
        return true;
    else
        return false;
}


void Yolo::drawPred(Mat& img, vector<Output> result, vector<Scalar> color)
{
    personnum = result.size();
    for (int i = 0; i < result.size(); i++)
    {
        int left, top;
        left = result[i].box.x;
        top = result[i].box.y;
        int color_num = i;
        rectangle(img, result[i].box, color[result[i].id], 2, 8);

        
        string label = className[result[i].id] + ":0." + to_string(int(result[i].confidence*100));
        Yolo::emotion[result[i].id] += 1;

        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
    }
    /*使用提取的目标框结果对图像进行绘制和标记，以及对每个目标框的类别进行统计：
for (int i = 0; i < result.size(); i++)：对于每个提取的目标框，循环迭代。
int left, top; left = result[i].box.x; top = result[i].box.y;：获取当前目标框的左上角坐标。
int color_num = i;：为目标框选择一个颜色编号。
rectangle(img, result[i].box, color[result[i].id], 2, 8);：在图像上绘制目标框，使用指定颜色和线宽。
string label = className[result[i].id] + ":0." + to_string(int(result[i].confidence*100));：构建目标框的标签，包括类别名称和置信度。
Yolo::emotion[result[i].id] += 1;：对目标框的类别进行统计，将类别的计数加1。
int baseLine; Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);：计算标签文本的大小。
top = max(top, labelSize.height);：确保标签绘制在目标框上方。
putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);：在图像上绘制标签文本，指定位置、字体、颜色和线宽。
将提取的目标框绘制在图像上，并在每个目标框上方添加类别标签。同时，它还对每个类别的目标框进行计数统计，以便后续分析和处理。*/

    if (window) {
        imshow("emotion", img);
        checkwindow = 0;
    }
    else if(checkwindow==0){
        cv::destroyWindow("emotion");
        checkwindow = 1;
    };
}
