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
	vector<Vec4f> lines;
	lineDetector(strSimpleBlobDetector, lines);
	cout << "finish !" << endl;

	return 0;
}


