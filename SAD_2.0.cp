 //
//  main.cpp
//  opencv0402
//
//  Created by 刘嘉诚 on 2021/4/2. SGBM算法
//
//Users/user/Desktop/left0/l30.bmp
//Users/user/Desktop/right0/r30.bmp
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <iostream>

//******************SAD************************
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

//定义图片读取位置
//string file_dir = "C:\\Program Files\\FLIR Integrated Imaging Solutions\\Triclops Stereo Vision SDK\\stereomatching\\Grab_Stereo\\pictures\\";
//定义图片存储位置
//string save_dir = "C:\\Program Files\\FLIR Integrated Imaging Solutions\\Triclops Stereo Vision SDK\\stereomatching\\Grab_Stereo\\";
//-------------------定义Sad处理图像函数---------------------
//int sub_kernel(Mat &kernel_left, Mat &kernel_right);
//Mat Left_sad, Right_sad;

//--------------------得到Disparity图像-----------------------
int winSize = 3;//匹配窗口的大小   //可以改变大小
float sub_Sum;//存储匹配范围的视差和。声明为float类型是因为一个像素为8位的uchar类型，像素差加起来要比8位多，float是32位

int DSR = 10;//视差搜索范围   //可以改变大小
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat getDisparity(Mat &left, Mat &right);//函数声明 以Mat类型返回

int main()
{
    //-----------------显示左右灰度图像---------------
    rgbImageL = imread("/Users/user/Desktop/left0/l27.bmp", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
    rgbImageR = imread("/Users/user/Desktop/right0/r27.bmp", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
    Mat Disparity;
    namedWindow("Renwu_left", 1);
    namedWindow("Renwu_right", 1);
    imshow("Renwu_left", rgbImageL);
    waitKey(5);
    imshow("Renwu_right", rgbImageR);
    waitKey(5);

    //------------------处理左右图得到视差图像---------
    Disparity = getDisparity(rgbImageL, rgbImageR);
    namedWindow("Disparity", 1);
    imshow("Disparity", Disparity);
    waitKey(0);

    return 0;
}

int sub_kernel(Mat &kernel_left, Mat &kernel_right)
{
    Mat Dif;
    absdiff(kernel_left, kernel_right, Dif);
    Scalar Add ;
    Add = sum(Dif);
    sub_Sum = Add[0];
    return sub_Sum;//返回匹配窗像素相减之后的和
}

Mat getDisparity(Mat &left, Mat &right)
{
    double start, end;//定义开始处理图像的时间和结束时间
    start = getTickCount();
    int ImgHeight = left.rows;
    int ImgWidth = left.cols;

    //------------------处理图像kernel大小--------------
    Mat Kernel_L(Size(winSize, winSize), CV_8UC1, Scalar::all(0));
    Mat Kernel_R(Size(winSize, winSize), CV_8UC1, Scalar::all(0));
    Mat disparity(ImgHeight, ImgWidth, CV_8UC1,Scalar(0));//视差图


    for (int i = 0; i < ImgHeight - winSize; i++)
    {
        for (int j = 0; j < ImgWidth - winSize; j++)
        {
            Kernel_L = left(Rect(j, i, winSize, winSize));
            Mat Temp(1, DSR, CV_32F, Scalar(0));//之所以是float型，是因为求取两个图像的差之后相加得到的和可能会大于8位
            //------------------将左右图视差放入matchLevel数组中-----------
            for (int k = 0; k < DSR; k++)
            {
                int y = j - k;
                if (y >= 0)
                {
                    Kernel_R = right(Rect(y, i, winSize, winSize));
                    //---------------对左右图kernel进行处理---------------
                    Temp.at<float>(k) = sub_kernel(Kernel_L, Kernel_R);
                }
            }
            //---------------寻找最佳匹配点--------------
            Point minLoc;
            minMaxLoc(Temp, NULL, NULL, &minLoc, NULL);

            int loc = minLoc.x;//之所以是x坐标，请参考我文中的解释
            disparity.at<uchar>(i, j) = loc * 16;
        }
    }
    //记录运行 时间
    end = getTickCount();
    double time = ((end - start)/CLOCKS_PER_SEC);
    cout << "Time is : " << time << "ms" << endl;
    return disparity;
}
