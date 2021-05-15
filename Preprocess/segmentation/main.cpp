/*==============================================================================================
Program: Segmentation
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2020/5/13
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2020-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include "SGSmooth.hpp"

using namespace std;
using namespace cv;

void findRPE(const cv::Mat& inImg, vector<int>& RPE){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 3);
    // parameters
    int N = row/2 - 50; // vertical ranges
    int p = 1;   // threshold of jump pixels
    double lamda = 0; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (fltImg.at<uchar>(i+N+row/2-N, 1) + fltImg.at<uchar>(i+N+k+row/2-N, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - fltImg.at<uchar>(i+N+row/2-N, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    RPE[column-1] = i_min[column-1] - 1 + row/2 - N;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    RPE[column-2] = i_min[column-2] - N - 1 + row/2;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        RPE[n-1] = i_min[n-1] - N - 1 + row/2;
    }
}

void findILM(const cv::Mat& inImg, vector<int>& RPE, vector<int>& ILM){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 11);
    Mat YGrad, YGradientABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    for (int i = 0; i < YGrad.rows; i++) {
        char * p = YGrad.ptr<char>(i);
        for (int j = 0; j < YGrad.cols; j++) {
            if (p[j] < 0) p[j] = 0;
        }
    }
    convertScaleAbs(YGrad, YGradientABS);
    for (int j = 0; j < column; j++){
        for (int i = RPE[j]-30; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < column; j++){
        for (int i = 0; i < RPE[j]-120; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YGradientABS.png", YGradientABS);
//    imshow("YGradientABS2", YGradientABS);
//    waitKey();

    int RPELoc = *max_element(RPE.begin(),RPE.end());
    // parameters
    int N = (RPELoc - 30 - 50)/2; // vertical ranges: 2*N+1
    int p = 3;   // threshold of jump pixels
    double lamda = 10; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YGradientABS.at<uchar>(i+N+50, 1) + YGradientABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YGradientABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    ILM[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    ILM[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        ILM[n-1] = i_min[n-1] - 1 + 50;
    }
}

void findONL(const cv::Mat& inImg, vector<int>& RPE, vector<int>& ONL){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 7);
    Mat YGrad, YGradientABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    convertScaleAbs(YGrad, YGradientABS);
    for (int j = 0; j < column; j++){
        for (int i = RPE[j]-6; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < column; j++){
        for (int i = 0; i < RPE[j] - 20; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YGradientABS.png", YGradientABS);
//    imshow("YGradientABS", YGradientABS);
//    waitKey();

    int RPELoc = *max_element(RPE.begin(),RPE.end());
    // parameters
    int N = (RPELoc - 50)/2; // vertical ranges: 2*N+1
    int p = 3;   // threshold of jump pixels
    double lamda = 50; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YGradientABS.at<uchar>(i+N+50, 1) + YGradientABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YGradientABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    ONL[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    ONL[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        ONL[n-1] = i_min[n-1] - 1 + 50;
    }
}

int main() {
    //Mat image = imread("/Volumes/Adam_U/image/original/29827_13.png", IMREAD_GRAYSCALE);
//    Mat image1 = imread("/Users/adam_lau/Downloads/4.jpg", IMREAD_GRAYSCALE);
//    Mat image2;
//    resize(image1, image2, cv::Size(340, 340),  0, 0, INTER_LINEAR);
//    imshow("sd", image2);
//    waitKey(0);
//    imwrite("/Users/adam_lau/Downloads/4.bmp", image2);

    Mat image = imread("2.jpg", IMREAD_GRAYSCALE);
    Mat seg = image.clone();

    int row = image.rows;
    int column = image.cols;

    vector <int> layerRPE(column);
    findRPE(image, layerRPE);
    vector <int> layerILM(column);
    findILM(image, layerRPE, layerILM);
    vector <int> layerONL(column);
    findONL(image, layerRPE, layerONL);

    for (int n = 0; n < column; ++n) {
        if ((layerRPE[n] - layerONL[n]) < 0) layerRPE[n] = layerONL[n];
    }
    vector<double> layerILM_double(layerILM.begin(), layerILM.end());
    vector<double> layerONL_double(layerONL.begin(), layerONL.end());
    vector<double> layerRPE_double(layerRPE.begin(), layerRPE.end());

    layerILM_double = sg_smooth(layerILM_double, 5, 0);
    layerONL_double = sg_smooth(layerONL_double, 5, 0);
    layerRPE_double = sg_smooth(layerRPE_double, 5, 0);

    cvtColor(image, image, COLOR_GRAY2BGR);
    for(int n = 0; n < image.cols; n++) {
        Point ilm(n, layerILM_double[n]);
        circle(image, ilm, 1, Scalar(0, 0, 255), -1); //红色
        Point onl(n, layerONL_double[n]);
        circle(image, onl, 1, Scalar(255, 245, 0), -1); //淡蓝
        Point rpe(n, layerRPE_double[n]);
        circle(image, rpe, 1, Scalar(0, 255, 255), -1); //黄色
    }
    imshow("segmentation", image);
    waitKey();
    imwrite("segmentated.jpg", image);

    for (int j = 0; j < column; j++){
        for (int i = layerONL_double[j]; i < row; i++){
            seg.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < column; j++){
        for (int i = 0; i < layerILM_double[j]; i++){
            seg.at<uchar>(i,j) = 0;
        }
    }
    imshow("1", seg);
    waitKey();
    imwrite("222.jpg", seg);

    return 0;
}


