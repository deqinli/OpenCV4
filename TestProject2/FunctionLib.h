#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <halconcpp\HalconCpp.h>
#include <map>
#include <cmath>
#include <numeric>
#include <QtCore\qstring.h>



typedef unsigned char uchar;

using namespace std;
using namespace cv;
using namespace HalconCpp;


class CFunctionLib
{
public:
	CFunctionLib();
	~CFunctionLib();
	// 计算完整瓷砖的Lab通道均值
	bool CalculateMeanValueOfWholeTiles();
	

	//获取色号合并后的输出等级
	bool GetCombinedResultGrade(map<int, vector<int>> map_CombineColorModel, const int nOriGrade, int &nResultGrade);


private:
	// 定义常量参数
	const float param_1_Above_3 = 1.0f / 3.0f;
	const float param_16_Above_116 = 16.0f / 116.0f;
	const float Xn = 0.950456f;
	const float Yn = 1.0f;
	const float Zn = 1.088754f;
	const float tValue = pow((6.0f / 29.0f), 3);
	const float coeffValue = (29.0f * 29.0f) / (3.0f * 6.0f * 6.0f);

public:
	float Gamma(float value);
	bool RGB2Lab(Mat &m_SrcImage, vector<Mat>& vLabImg);
	bool RGB2Lab(uchar rSource, uchar gSource, uchar bSource, float &Lab_L, float &Lab_a, float &Lab_b);
	bool calcColorDiff(Mat srcImage, Point point1, Point point2, double &dDiffValue);
	bool calcColorDiff(Mat srcImage, Point point1, Point point2, double &dDiffValue, std::string mode);
	bool SeparateTilesFromSrcImage(HObject ho_srcImage, HObject &ho_resultImage, map<string, int> KV_Map);
	bool ImageEqualDivided(Mat srcImage, vector<Mat> &vec_SubImages, int subImageNum = 3);
	bool MeanValueOfLab(Mat &srcImage, vector<float> &MeanValueOfLab_Lab);
	bool MeanValueOfLab(Mat &srcImage, Scalar &mean, Scalar &std_value);
	bool MaxDistance(vector<vector<float>> vec_LabPoint, double &dMaxDistance);
	Mat HObject2Mat(HObject hoImage);

	//bool MyWritePrivateProfileInt(QString strApp, QString strKey, int iContent, QString strFilePath);
};

namespace FunctionLib {

struct CmpByValueUp
	{
		bool operator()(const tuple<int, double, double>& lhs, const tuple<int, double, double>& rhs)
		{
			//从小到大排序
			return get<2>(lhs) < get<2>(rhs);
		}
	};

struct CmpByValueDown
	{
		bool operator()(const tuple<int, double, double>& lhs, const tuple<int, double, double>& rhs)
		{
			//从大到小排序
			return get<2>(lhs) > get<2>(rhs);
		}
	};

bool ReadIniFile(const QString path);
bool WriteIniFile(const QString path);
//void on_CornerHarris(int, void*, Mat g_srcImage,Mat g_srcImage1);

//sift 特征检测和定位：位置配准
void Sift_test(string imgPath1, string imgPath2);

//斑点检测
void BlobDectector(string imgPath);

//毛刺检测
void GlitchDetector(string imgPath);

//划痕检测
void surface_scratch(string imgPath);

//角点检测
void cornerHarris(string imgPath);

//圆形拟合
bool fitCircle(string imgPath);

//曲线拟合
bool fitCurve(string imgPath);

//直线拟合
bool fitLine(string imgPath);

//多项式曲线拟合
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);

//直线检测
bool lineDetector(string imgPath, vector<Vec4f> &lines);

//计算两直线交点
Point2f GetLineCrossPoint(Vec4i LineA,Vec4i LineB);

//计算多条直线两两之间的交点
bool GetMultyLineCrossPoint(vector<Vec4f> InputLines, vector<Point2f> &OutputPointList);

//在图片上画点
bool DrawPoint(vector<Point2f> InputPointList, Mat &DrawImage);

//计算点到直线的距离
bool Distance_PointToLine(Vec4i Line, Point P, double &Distance);

//计算两条直线的夹角
bool GetAngleOfTwoLines(Vec4i InputLine_1, Vec4i InputLine_2, double &OutputdAngle, bool InputFlagDegree = 1);

//计算两点中心
bool GetmiddlePoint(Point Input_P1, Point Input_P2, Point OutputMiddlePoint);

//获取旋转矩阵
bool GetRotation2DMatrix(Vec4i InputLine, Point InputCenterPoint, double InputDoubleAngle, bool InputBoolDirection);


//旋转直线
bool GetRotatePoints(vector<Point> InputPoints, vector<Point2d> &OutputPoints, Point InputCenterPoint, double InputDoubleAngle, bool InputBoolDirection);

// 旋转直线test
bool RotateTest();

void rotatePoint(double angle, Point &rotate_pt, Point origin_pt, Point center_pt);


}

