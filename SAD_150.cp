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

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
using namespace std;
using namespace cv;

const int n = 7; //窗口大小：2*n+1
const int range = 150;//视差范围

void SAD(uchar* Limg,
    uchar* Rimg,
    uchar* Oimg,int w,int h)
{
    for (int y = 0; y< h; y++)
    {
        for (int x = 0; x< w; x++){
            unsigned int bestCost = 999999;
            unsigned int bestDisparity = 0;
            for (int d = 0; d <= range; d++)
            {
                unsigned int cost = 0;
                for (int i = -n; i <= n; i++)
                {
                    for (int j = -n; j <= n; j++)
                    {
                        int yy, xx, xxd;
                        yy = y + i;
                        if (yy < 0) yy = 0;
                        if (yy >= h) yy = h-1;

                        xx = x + j;
                        if (xx < 0) xx = 0;
                        if (xx >= w) xx = w-1;

                        xxd = xx - d;
                        if (xxd < 0) xxd = 0;
                        if (xxd >= w) xxd = w-1;
                        cost += abs((int)(Limg[yy*w + xx] - Rimg[yy*w + xxd]));
                    }
                }
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestDisparity = d;
                }
                Oimg[y*w + x] = bestDisparity*4;
            }
        }
    }
}

int main()
{
    clock_t starttime = clock();
    Mat imL, imR, imO;
    imL = imread("/Users/user/Desktop/left0/l27.bmp", 0);
    if (imL.empty())
    {
        return -1;
    }
    imR = imread("/Users/user/Desktop/right0/r27.bmp", 0);
    if (imR.empty())
    {
        return -1;
    }
    imO.create(imL.rows, imL.cols, CV_8UC1);

    SAD(imL.data, imR.data, imO.data, imL.cols, imL.rows);
    namedWindow("left", WINDOW_AUTOSIZE);
    namedWindow("right", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imwrite("11.png", imO);
    imshow("Output", imO);
    imshow("left", imL);
    imshow("right", imR);
    clock_t endtime = clock();
    printf("%d\n", (endtime - starttime));
    cout << (endtime - starttime) << endl;
    waitKey(0);
    return 0;
}



