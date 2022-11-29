#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main(){
    cout << "[ onnxファイルとopencvを使って推論します ]" << endl;

    auto model="./mask_detect.onnx";
    dnn::Net net=dnn::readNet(model);
    cout << "=========onnxファイルの読み込み完了=========" << endl;

    Mat img=imread("./testdata/1175.jpg");
    Mat blob=dnn::blobFromImage(img, 1.0/255);
    net.setInput(blob, "input");
    Mat prob=net.forward("output");
    prob=prob.reshape(1,1);
    cout << "=========推論完了=========" << endl;
    cout << prob << endl;



    return 0;
}
