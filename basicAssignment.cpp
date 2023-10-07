#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<vector>
#include<cmath>
#include<iostream>

using namespace std;
using namespace cv;

#ifndef _DETECTOR_H_
#define _DETECTOR_H_

class ArmorDetect
{
    
    public:
    Mat preprocess(Mat img);
    vector<RotatedRect> Findcounters(Mat pre);
    vector<RotatedRect> Armordetect(vector<RotatedRect> light);
    vector<Point2f> point_2d(Mat img, RotatedRect ve);
    vector<Point3f> point_3d();

    Mat cameraMatrix = (Mat_<double>(3, 3) << 1.2853517927598091e+03, 0., 3.1944768628958542e+02, 0.,
       1.2792339468697937e+03, 2.3929354061292258e+02, 0., 0., 1.);
    Mat distCoeffs = (Mat_<double>(1, 5) << -6.3687295852461456e-01, -1.9748008790347320e+00,
       3.0970703651800782e-02, 2.1944646842516919e-03, 0.);
    Mat rvec, tvec;
    vector<Point2f> Points2d;
    vector<Point3f> Points3d;
};

#endif

Mat ArmorDetect::preprocess(Mat img)
{
     Mat cameraMatrix = (Mat_<double>(3, 3) << 1.2853517927598091e+03, 0., 3.1944768628958542e+02, 0.,
       1.2792339468697937e+03, 2.3929354061292258e+02, 0., 0., 1.);
    Mat distCoeffs = (Mat_<double>(1, 5) << -6.3687295852461456e-01, -1.9748008790347320e+00,
       3.0970703651800782e-02, 2.1944646842516919e-03, 0.);
    Mat img_clone;

     undistort(img, img_clone, cameraMatrix, distCoeffs);

     vector<Mat> channels;
     split(img_clone, channels);

     Mat img_gray;
     img_gray = channels.at(2);
     int value = 230;
     Mat dst;
     threshold(img_gray, dst, value, 255, THRESH_BINARY);
     
     Mat pre;
     Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
    morphologyEx(dst, pre, MORPH_OPEN, kernel); 
    return pre;
}

vector<RotatedRect> ArmorDetect::Findcounters(Mat pre)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(pre, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));

    vector<RotatedRect> light;
    for (size_t i = 0; i < contours.size(); ++i)
    {
 
        RotatedRect light_rect = minAreaRect(contours[i]);
        double width = light_rect.size.width; 
        double height = light_rect.size.height; 
        double area = width * height; 
        if (area < 10)
            continue;
        if (width / height > 0.4)
            continue;
        else
            light.push_back(light_rect);
    }
    return light;
}

vector<RotatedRect> ArmorDetect::Armordetect(vector<RotatedRect> light)
{
    vector<RotatedRect> armor_final;
    RotatedRect armor; 
    double angle_differ; 

    if (light.size() < 2)
        return armor_final;

    for (size_t i = 0; i < light.size() - 1; i++)
    {
        for (size_t j = i+1; j < light.size(); j++)
        {
            angle_differ = abs(light[i].angle - light[j].angle);
 
            bool if1 = (angle_differ < 8); 
            if (if1)
            {
                armor.center.x = (light[i].center.x + light[j].center.x) / 2;
                armor.center.y = (light[i].center.y + light[j].center.y) / 2; 
                armor.angle = (light[i].angle + light[j].angle) / 2;
                armor.size.width = abs(light[i].center.x - light[j].center.x); 
                armor.size.height = light[i].size.height;
                armor_final.push_back(armor);
            }
        }
    }
    return armor_final;
}

vector<Point2f> ArmorDetect::point_2d(Mat img, RotatedRect ve)
{
    Point2f pt[4];
    vector<Point2f> point2d;
    int i;
    
    for (i = 0; i < 4; i++)
    {
        pt[i].x = 0;
        pt[i].y = 0;
    }
    ve.points(pt); 

    line(img, pt[0], pt[2], Scalar(0, 0, 255), 2, 4, 0);  
    line(img, pt[1], pt[3], Scalar(0, 0, 255), 2, 4, 0);  
    line(img, pt[0], pt[1], Scalar(0, 0, 255), 2, 4, 0);  
    line(img, pt[1], pt[2], Scalar(0, 0, 255), 2, 4, 0);  
    line(img, pt[2], pt[3], Scalar(0, 0, 255), 2, 4, 0); 
    line(img, pt[3], pt[0], Scalar(0, 0, 255), 2, 4, 0); 

    point2d.push_back(pt[0]);
    point2d.push_back(pt[1]);
    point2d.push_back(pt[2]);
    point2d.push_back(pt[3]);
    return point2d;           
}

vector<Point3f> ArmorDetect:: point_3d()
{
    vector<Point3f> Points3d;
    Point3f point3f;
    
    point3f.x = 0;
    point3f.y = 0;
    point3f.z = 0;
    Points3d.push_back(point3f);
    
    point3f.x = 0;
    point3f.y = 5.2;
    point3f.z = 0;
    Points3d.push_back(point3f);
    
    point3f.x = 13.8;
    point3f.y = 5.2;
    point3f.z = 0.0;
    Points3d.push_back(point3f);
    
    point3f.x = 13.8;
    point3f.y = 0;
    point3f.z = 0;
    Points3d.push_back(point3f);
    return Points3d;
}

int main()
{
    vector<RotatedRect> light;
    vector<RotatedRect> armor_final;
    ArmorDetect detect; 

    string path="../Video/装甲板.avi";
    VideoCapture cap(path);
    Mat img;
    while (true){
    cap.read(img);
    Mat img_clone = img.clone();
    Mat pre = detect.preprocess(img_clone);
    light = detect.Findcounters(pre);
    armor_final = detect.Armordetect(light);
     for (size_t i = 0; i < armor_final.size(); i++)
    {
        detect.Points2d = detect.point_2d(img, armor_final[i]);
    }
     detect.Points3d = detect.point_3d();

     solvePnP(detect.Points3d, detect.Points2d, detect.cameraMatrix, detect.distCoeffs, detect.rvec, detect.tvec, false, SOLVEPNP_ITERATIVE);
     string d = to_string(detect.tvec.at<double>(2, 0)) + "cm";
     putText(img, d, detect.Points2d[2], 2, 1, Scalar(18, 195, 127));

     imshow("out", img);
    waitKey(30);
    }
    return 0;
}