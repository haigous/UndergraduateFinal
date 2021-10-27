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

//*************************Census*********************
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

//-------------------定义汉明距离----------------------------
int disparity;
int GetHammingWeight(uchar value);//求1的个数

//-------------------定义Census处理图像函数---------------------
int hWind = 1;//定义窗口大小为（2*hWind+1）
Mat ProcessImg(Mat &Img);//将矩形内的像素与中心像素相比较，将结果存于中心像素中
Mat Img_census, Left_census, Right_census;

//--------------------得到Disparity图像------------------------
Mat getDisparity(Mat &left, Mat &right);

//--------------------处理Disparity图像-----------------------
Mat ProcessDisparity(Mat &disImg);

int ImgHeight, ImgWidth;

//int num = 0;//异或得到的海明距离
Mat LeftImg, RightImg;
Mat DisparityImg(ImgHeight, ImgWidth, CV_8UC1, Scalar::all(0));
Mat DisparityImg_Processed(ImgHeight, ImgWidth, CV_8UC1, Scalar::all(0));
Mat DisparityImg_Processed_2(ImgHeight, ImgWidth, CV_8UC1);
//定义读取图片的路径
//string file_dir="C:\\Program Files\\FLIR Integrated Imaging Solutions\\Triclops Stereo Vision SDK\\stereomatching\\Grab_Stereo\\pictures\\";
//定义存储图片的路径
//string save_dir= "C:\\Program Files\\FLIR Integrated Imaging Solutions\\Triclops Stereo Vision SDK\\stereomatching\\Grab_Stereo\\Census\\";

int main()
{
    LeftImg = imread("/Users/user/Desktop/left0/l27.bmp", 0);
    RightImg = imread("/Users/user/Desktop/right0/r27.bmp", 0);
    namedWindow("renwu_left", 1);
    namedWindow("renwu_right", 1);
    imshow("renwu_left", LeftImg);
    waitKey(5);
    imshow("renwu_right", RightImg);
    waitKey(5);
    ImgHeight = LeftImg.rows;
    ImgWidth = LeftImg.cols;

    Left_census= ProcessImg(LeftImg);//处理左图，得到左图的CENSUS图像 Left_census
    namedWindow("Left_census", 1);
    imshow("Left_census", Left_census);
    waitKey(5);
//  imwrite(save_dir + "renwu_left.jpg", Left_census);

    Right_census= ProcessImg(RightImg);
    namedWindow("Right_census", 1);
    imshow("Right_census", Right_census);
    waitKey(5);
//  imwrite(save_dir  + "renwu_right.jpg", Right_census);

    DisparityImg= getDisparity(Left_census, Right_census);
    namedWindow("Disparity", 1);
    imshow("Disparity", DisparityImg);
//  imwrite(save_dir  + "disparity.jpg", DisparityImg);
    waitKey(5);

    DisparityImg_Processed = ProcessDisparity(DisparityImg);
    namedWindow("DisparityImg_Processed", 1);
    imshow("DisparityImg_Processed", DisparityImg_Processed);
//  imwrite(save_dir + "disparity_processed.jpg", DisparityImg_Processed);
    waitKey(0);
    return 0;
}


//-----------------------对图像进行census编码---------------
Mat ProcessImg(Mat &Img)
{
    int64 start, end;
    start = getTickCount();

    Mat Img_census = Mat(Img.rows, Img.cols, CV_8UC1, Scalar::all(0));
    uchar center = 0;

    for (int i = 0; i < ImgHeight - hWind; i++)
    {
        for (int j = 0; j < ImgWidth - hWind; j++)
        {
            center = Img.at<uchar>(i + hWind, j + hWind);
            uchar census = 0;
            uchar neighbor = 0;
            for (int p = i; p <= i + 2 * hWind; p++)//行
            {
                for (int q = j; q <= j + 2 * hWind; q++)//列
                {
                    if (p >= 0 && p <ImgHeight  && q >= 0 && q < ImgWidth)
                    {

                        if (!(p == i + hWind && q == j + hWind))
                        {
                            //--------- 将二进制数存在变量中-----
                            neighbor = Img.at<uchar>(p, q);

                            if (neighbor > center)
                            {
                                census = census * 2;//向左移一位，相当于在二进制后面增添0
                            }
                            else
                            {
                                census = census * 2 + 1;//向左移一位并加一，相当于在二进制后面增添1
                            }
                            //cout << "census = " << static_cast<int>(census) << endl;
                        }
                    }
                }

            }
            Img_census.at<uchar>(i + hWind, j + hWind) = census;
        }
    }
    /*end = getTickCount();
    cout << "time is = " << end - start << " ms" << endl;*/
    return Img_census;
}

//------------得到汉明距离---------------
int GetHammingWeight( uchar value)
{
    int num = 0;
    if (value == 0)
        return 0;
    while (value)
    {
        ++num;
        value = (value - 1)&value;
    }
    return num;
}

//--------------------得到视差图像--------------
Mat getDisparity(Mat &left, Mat &right)
{
    int DSR =16;//视差搜索范围
    Mat disparity(ImgHeight,ImgWidth,CV_8UC1);

    cout << "ImgHeight = " << ImgHeight << "   " << "ImgWidth = " << ImgWidth << endl;
    for (int i = 0; i < ImgHeight; i++)
    {
        for (int j = 0; j < ImgWidth; j++)
        {
            uchar L;
            uchar R;
            uchar diff;

            L = left.at<uchar>(i, j);
            Mat Dif(1, DSR, CV_8UC1);
//          Mat Dif(1, DSR, CV_32F);

            for (int k = 0; k < DSR; k++)
            {
                //cout << "k = " << k << endl;
                int y = j - k;
                if (y < 0)
                {
                    Dif.at<uchar>(k) = 0;
                }
                if (y >= 0)
                {
                    R = right.at<uchar>(i,y);
                    //bitwise_xor(L, R, );
                    diff = L^R;
                    diff = GetHammingWeight(diff);
                    Dif.at<uchar>(k) = diff;
//                  Dif.at<float>(k) = diff;
                }
            }
            //---------------寻找最佳匹配点--------------
            Point minLoc;
            minMaxLoc(Dif, NULL, NULL, &minLoc, NULL);
            int loc = minLoc.x;
            //cout << "loc..... = " << loc << endl;
            disparity.at<uchar>(i,j)=loc*16;
        }
    }
    return disparity;
}

//-------------对得到的视差图进行处理-------------------
Mat ProcessDisparity(Mat &disImg)
{
    Mat ProcessDisImg(ImgHeight,ImgWidth,CV_8UC1);//存储处理后视差图
    for (int i = 0; i < ImgHeight; i++)
    {
        for (int j = 0; j < ImgWidth; j++)
        {
            uchar pixel = disImg.at<uchar>(i, j);
            if (pixel < 100)
                pixel = 0;
            ProcessDisImg.at<uchar>(i, j) = pixel;
        }
    }
    return ProcessDisImg;
}
