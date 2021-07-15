#include <iostream>
#include <numeric>
#include <vector>
#include "FunctionLib.h"
#include <fstream>

using namespace std;
using namespace FunctionLib;


int main()
{
	string strSimpleBlobDetector = "E:/SimpleBlobDetector.bmp";
	//string hello2 = "E:/hello2.jpg";
	//
	//string fin1 = "E:/TestImage20210615/fin1.png";
	//string surface_scrach = "E:/TestImage20210615/surface_scrach.png";
	//BlobDectector(strSimpleBlobDetector);
	//surface_scratch(surface_scrach);

	/*string imgPath = "E:/1-work/code/code_python/opencvTest/image/FitCircle_1_20210701.jpg";
	fitCurve(imgPath);*/
	//Sift_test(fin1,fin1);
	//GlitchDetector(strSimpleBlobDetector);
	//cornerHarris(strSimpleBlobDetector);

	//fitLine(strSimpleBlobDetector);
	//fitCircle(strSimpleBlobDetector);
	//vector<Vec4f> lines;
	//lineDetector(strSimpleBlobDetector, lines);


	//RotateTest();

	vector<Point> vec_Point;
	Point p1 = Point(100, 100);
	Point p2 = Point(100, 300);
	Point p3 = Point(300, 300);
	Point p4 = Point(300, 100);

	vec_Point.push_back(p1);
	vec_Point.push_back(p2);
	vec_Point.push_back(p3);
	vec_Point.push_back(p4);

	Mat img = Mat::zeros(1000,1000, CV_8UC3);

	vector<Point2d> outputPoint;
	GetRotatePoints(vec_Point, outputPoint, Point(500,500), -PI/2, 0);
	line(img, p1, p2, Scalar(0, 0, 255), 3);
	line(img, p2, p3, Scalar(0, 0, 255), 3);
	line(img, p3, p4, Scalar(0, 0, 255), 3);
	line(img, p4, p1, Scalar(0, 0, 255), 3);
	line(img, outputPoint[0], outputPoint[1], Scalar(255, 255, 255), 3);
	line(img, outputPoint[1], outputPoint[2], Scalar(255, 255, 255), 3);
	line(img, outputPoint[2], outputPoint[3], Scalar(255, 255, 255), 3);
	line(img, outputPoint[3], outputPoint[0], Scalar(255, 255, 255), 3);

	//cout << outputPoint << endl;

	imshow("hello", img);
	waitKey(0);




	cout << "finish !" << endl;

	return 0;
}


