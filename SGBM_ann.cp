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

const int imageWidth = 640;                             //摄像头的分辨率
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);         //图像尺寸

Mat rgbImageL, grayImageL;           //Mat类是用于保存图像以及其他矩阵数据的数据结构
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;            //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域
Rect validROIR;            //rect矩形函数

Mat mapLx, mapLy, mapRx, mapRy;     //映射表
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象


Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
//对应matlab里的左相机标定矩阵
Mat cameraMatrixL = (Mat_<double>(3, 3) << 1123.507738058886, 0, 324.7238859005240,
    0, 1127.282945351939, 263.3336974194803,
    0, 0, 1.0);
//对应Matlab所得左相机畸变参数
Mat distCoeffL = (Mat_<double>(5, 1) << -0.109013556867451, 0.709661003534675, 0.002319272467005, -0.0009934357148160210, -4.116358718594088);  //（K1，K2，P1，P2，K3）

//对应matlab里的右相机标定矩阵
Mat cameraMatrixR = (Mat_<double>(3, 3) << 1125.761625967723, 0, 319.3887168539959,
    0, 1129.305978447782, 270.6640131815286,
    0, 0, 1.0);
//对应Matlab所得右相机畸变参数
Mat distCoeffR = (Mat_<double>(5, 1) << -0.136891100359016, 1.064612859061886, 0.001689182346785,-0.0008868523127893606, -5.963649858150998);  //（K1，K2，P1，P2，K3）
 
Mat T = (Mat_<double>(3, 1) << -120.6912133121152, 0.855744633796061, 2.316970609061114);       //T平移向量
//Mat rec = (Mat_<double>(3, 1) <<-0.00968, 0.12232, -0.01482);//rec旋转向量
//Mat R;//R 旋转矩阵

Mat R = (Mat_<double>(3, 3) << 0.999923155739635, -0.001866975165498, 0.012255489358735,
         0.001942204577353, 0.999979328954652, -0.006129396771595,
         -0.012243792593407, 0.006152728430164, 0.999906112330450);


static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 16.0e4;
    FILE* fp = fopen(filename, "wt");
    printf("%d %d \n", mat.rows, mat.cols);
    for (int y = 0; y < mat.rows; y++)
    {
        for (int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);

        }
    }
    fclose(fp);
}

/*给深度图上色*/
void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
{
    // color map
    float max_val = 255.0f;
    float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
    { 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
    float sum = 0;
    for (int i = 0; i<8; i++)
        sum += map[i][3];

    float weights[8]; // relative   weights
    float cumsum[8];  // cumulative weights
    cumsum[0] = 0;
    for (int i = 0; i<7; i++) {
        weights[i] = sum / map[i][3];
        cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
    }

    int height_ = src.rows;
    int width_ = src.cols;
    // for all pixels do
    for (int v = 0; v<height_; v++) {
        for (int u = 0; u<width_; u++) {

            // get normalized value
            float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);

            // find bin
            int i;
            for (i = 0; i<7; i++)
                if (val<cumsum[i + 1])
                    break;

            // compute red/green/blue values
            float   w = 1.0 - (val - cumsum[i])*weights[i];
            uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
            uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
            uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
            //rgb内存连续存放
            disp.data[v*width_ * 3 + 3 * u + 0] = b;
            disp.data[v*width_ * 3 + 3 * u + 1] = g;
            disp.data[v*width_ * 3 + 3 * u + 2] = r;
        }
    }
}

      /*****立体匹配*****/
void stereo_match(int, void*)
{
    sgbm->setPreFilterCap(63);                       //水平sobel预处理后，映射滤波器大小；预过滤图像像素的截断值。默认为15
    int sgbmWinSize =  7;                            //匹配块大小，大于1的奇数，通常，它应该3～11范围内
    int NumDisparities = 240;                        //视差窗口(范围)，最大视差值和最小视差值之差，必须是16的倍数
    int UniquenessRatio = 12;                         //视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
    sgbm->setBlockSize(sgbmWinSize);
    int cn = rectifyImageL.channels();               //通道数

    sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);     //动态规划中P1 P2控制视差度光滑度 P1用于倾斜表面；一般：P1 = 8 * 通道数*SgbmWindowSize*SgbmWindowSize
    sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);    //P2用于“解决”深度不连续性问题，值越大，视差越平滑；通常P2 = 4 * P1，p1控制视差平滑度，p2值越大，差异越平滑
    sgbm->setMinDisparity(0);                            // 最小视差值
    sgbm->setNumDisparities(NumDisparities);             //视差窗口
    sgbm->setUniquenessRatio(UniquenessRatio);           //视差唯一性百分比
    sgbm->setSpeckleWindowSize(20);                     //平滑视差区域的最大尺寸，以考虑其噪声斑点并使其无效。
                                                         //将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
    sgbm->setSpeckleRange(2);                            // 视差变化阈值，每个连接组件内的最大视差变化。通常，1或2就足够好了
    sgbm->setDisp12MaxDiff(-1);                          // 左右视差图的最大容许差异（超过将被清零），默认为 - 1，即不执行左右视差检查。
    sgbm->setMode(StereoSGBM::MODE_SGBM);                //打开opencv自带SGBM算法文件
    
    Mat disp, dispf, disp8;
    sgbm->compute(rectifyImageL, rectifyImageR, disp);   //计算视差disp
    //去黑边
    Mat img1p, img2p;
    copyMakeBorder(rectifyImageL, img1p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
                           //（输入，输出，边界方向添加像素值，边界类型，边界值） BORDER_REPLICATE 重复：对边界像素进行复制
                           //扩充src原图的边缘，将图像变大，然后以各种外插方式自动填充图像边界
    copyMakeBorder(rectifyImageR, img2p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
    
    dispf = disp.colRange(NumDisparities, img2p.cols - NumDisparities);                      //colRange为指定的列span创建一个新的矩阵头，可取指定列区间元素；只取到左边界，不取右边界
    dispf.convertTo(disp8, CV_8U, 255 / (NumDisparities *16.));    //数据转换（转换成的目标矩阵，目标矩阵m的数据类型，缩放因子，增量）
    reprojectImageTo3D(dispf, xyz, Q, true);                       //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    imshow("disparity", disp8);
    Mat color(dispf.size(), CV_8UC3);
    GenerateFalseMap(disp8, color);  //调用GenerateFalseMap 转成彩图
    imshow("disparity", color);      //显示彩图
    saveXYZ("xyz.xls", xyz);
}



/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            break;
    }
}


/*****主函数*****/
int main()
{
    /*  立体校正    */
   // Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);                          //CALIB_ZERO_DISPARITY会让两幅校正后的图像的主点有相同的像素坐标
    //计算无畸变和修正转换关系，为了重映射，将结果以映射的形式表达
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);
    //（相机矩阵，畸变参数，修正变换矩阵，新的相机矩阵，未畸变图像尺寸，输出映射类型，第一个输出映射，第二个输出映射）
    /*  读取图片    */
    rgbImageL = imread("/Users/user/Desktop/left0/l26.bmp", CV_LOAD_IMAGE_COLOR);//CV_LOAD_IMAGE_COLOR
    rgbImageR = imread("/Users/user/Desktop/right0/r26.bmp", -1);


    /*  经过remap重映射之后，左右相机的图像已经共面并且行对准了 */
    remap(rgbImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR); //（输入，输出，存放图像X方向的映射关系，存放图像y方向的映射关系,插值方式）
    remap(rgbImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR); //INTER_LINEAR 双线性插值

    /*  把校正结果显示出来*/
    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 700. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

                                        //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分
    resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分
    resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);

    /*  立体匹配    */
    namedWindow("disparity", CV_WINDOW_NORMAL);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);//disparity 点击获取坐标
    stereo_match(0, 0);

    waitKey(0);
    return 0;
}

