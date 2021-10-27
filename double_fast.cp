//
//  main.cpp
//  opencv0402
//
//  Created by 刘嘉诚 on 2021/4/2. SGBM算法
//
// /Users/user/Desktop/left0/l30.bmp
// /Users/user/Desktop/right0/r30.bmp
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
int main()
{
    // load image
    Mat image = imread("/Users/user/Desktop/left0/l30.bmp",CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        printf("读取图片文件失败\n");
        exit(0);
    }
    //resize(image, image, Size(), 0.3, 0.3);
 
    imshow("salted image", image);
 
    //median filte
    Mat resutl;
    bilateralFilter(image, resutl, 45, 25 * 2, 25 / 2);
 
    //display result
    imshow("median filted image", resutl);
    waitKey();
}

