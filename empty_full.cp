//
//  main.cpp
//  opencv0402
//
//  Created by 刘嘉诚 on 2021/4/2. SGBM算法
//
// /Users/user/Desktop/left0/l30.bmp
// /Users/user/Desktop/right0/r30.bmp
#include "opencv2/imgcodecs/legacy/constants_c.h"
//
//  main.cpp
//  opencv0402
//
//  Created by 刘嘉诚 on 2021/4/2. SGBM算法
//
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// fillhole.cpp : 定义控制台应用程序的入口点。
//

 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
     

    #pragma comment(lib,"opencv_core2410d.lib")
    #pragma comment(lib,"opencv_highgui2410d.lib")
    #pragma comment(lib,"opencv_imgproc2410d.lib")
      
using namespace std;
using namespace cv;




void fillHole(const Mat srcBw, Mat &dstBw)
{
    Size m_Size = srcBw.size();
    Mat Temp=Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));

    Mat cutImg;//裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

int main()
{
Mat img=cv::imread("/Users/user/Desktop/double28_10.png");

Mat gray;
cv::cvtColor(img, gray, CV_RGB2GRAY);

Mat bw;
cv::threshold(gray, bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

Mat bwFill;
fillHole(bw, bwFill);

imshow("填充前", gray);
imshow("填充后", bwFill);
waitKey();
return 0;
}
