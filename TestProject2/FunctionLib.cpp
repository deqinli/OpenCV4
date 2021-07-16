#pragma execution_character_set("utf-8")
#include "FunctionLib.h"
#include <QtCore\qsettings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>
#include <QtCore\qtextcodec.h>
#include <QtCore\qdebug.h>
#include <opencv2\features2d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <time.h>
#include <math.h>

using namespace cv;
using namespace cv::ml;

using namespace std;



CFunctionLib::CFunctionLib()
{
}


CFunctionLib::~CFunctionLib()
{
}

float CFunctionLib::Gamma(float value)
{
	return value > 0.04045 ? powf((value + 0.055f) / 1.055f, 2.4f) : (value / 12.92);
}

bool CFunctionLib::RGB2Lab(Mat &m_SrcImg, vector<Mat> &vLabImg)
{
	vLabImg.clear();
	for (int i = 0; i < 3; i++)
	{
		vLabImg.push_back(Mat(m_SrcImg.size(), CV_64FC1, Scalar::all(0)));
	}
	for (int r = 0; r < m_SrcImg.rows; r++)
	{
		Vec3b *pVec3bSrc = m_SrcImg.ptr<Vec3b>(r);
		double *pVlabImg_L = vLabImg[0].ptr<double>(r);
		double *pVlabImg_a = vLabImg[1].ptr<double>(r);
		double *pVlabImg_b = vLabImg[2].ptr<double>(r);

		for (int c = 0; c < m_SrcImg.cols; c++)
		{
			int sR = (int)pVec3bSrc[c][0];
			int sG = (int)pVec3bSrc[c][1];
			int sB = (int)pVec3bSrc[c][2];

			double R = (double)sR / 255.0;
			double G = (double)sG / 255.0;
			double B = (double)sB / 255.0;

			double r, g, b;

			if (R <= 0.04045)
			{
				r = (double)R / 12.92;
			}
			else 
			{
				r = pow((R + 0.055) / 1.055, 2.4);
			}

			if (G <= 0.04045)
			{
				g = (double)G / 12.92;
			}
			else
			{
				g = pow((G + 0.055) / 1.055, 2.4);
			}

			if (B <= 0.04045)
			{
				b = (double)B / 12.92;
			}
			else
			{
				b = pow((B + 0.055) / 1.055, 2.4);
			}

			double X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
			double Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
			double Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;


			double epsilon = 0.008856;
			double kappa = 903.3;

			double Xr = 0.950456;
			double Yr = 1.0;
			double Zr = 1.088754;

			double xr = X / Xr;
			double yr = Y / Yr;
			double zr = Z / Zr;

			double fx, fy, fz;
			if (xr > epsilon)
			{
				fx = pow(xr, 1.0 / 3.0);
			}
			else
			{
				fx = (kappa * xr + 16) / 116.0;
			}

			if (yr > epsilon)
			{
				fy = pow(yr, 1.0 / 3.0);
			}
			else
			{
				fy = (kappa * yr + 16) / 116.0;
			}

			if (zr > epsilon)
			{
				fz = pow(zr, 1.0 / 3.0);
			}
			else
			{
				fz = (kappa * zr + 16) / 116.0;
			}

			pVlabImg_L[c] = 116.0 * fy - 16.0;
			pVlabImg_a[c] = 500.0 * (fx - fy);
			pVlabImg_b[c] = 200.0 * (fy - fz);
		}
	}
	return true;
}

bool CFunctionLib::RGB2Lab(uchar rSource, uchar gSource, uchar bSource, float &Lab_L, float &Lab_a, float &Lab_b)
{
	float fX, fY, fZ;

	float R = Gamma(rSource / 255.0);
	float G = Gamma(gSource / 255.0);
	float B = Gamma(bSource / 255.0);

	float X = 0.412453f * R + 0.357580f * G + 0.180423f * B;
	float Y = 0.212671f * R + 0.715160f * G + 0.072169f * B;
	float Z = 0.019334f * R + 0.119193f * G + 0.950227f * B;

	X /= (Xn);
	Y /= (Yn);
	Z /= (Zn);

	if (X > tValue)
	{
		fX = pow(X, param_1_Above_3);
	}
	else {
		fX = coeffValue * X + param_16_Above_116;
	}

	if (Y > tValue)
	{
		fY = pow(Y, param_1_Above_3);
	}
	else {
		fY = coeffValue * Y + param_16_Above_116;
	}

	if (Z > tValue)
	{
		fZ = pow(Z, param_1_Above_3);
	}
	else {
		fZ = coeffValue * Z + param_16_Above_116;
	}

	Lab_L = 116 * fY - 16;
	Lab_L = Lab_L > 0.0f ? Lab_L : 0.0f;
	Lab_a = 500 * (fX - fY);
	Lab_b = 200 * (fY - fZ);
	return true;
}

bool CFunctionLib::calcColorDiff(Mat srcImage, Point point1, Point point2, double &dDiffValue)
{
	std::string mode = "RGB";
	calcColorDiff(srcImage, point1, point2, dDiffValue, mode);
	return true;
}

bool CFunctionLib::calcColorDiff(Mat srcImage, Point point1, Point point2, double &dDiffValue, std::string mode)
{
	// ���ͼƬΪ�գ�����ֱ�ӷ���
	if (!srcImage.data)
	{
		return false;
	}


	cv::Mat Img = srcImage.clone();    // ����ԴͼƬ
	int b1, b2, g1, g2, r1, r2;        // �����������rgbͨ��ֵ��
	float Lab_L1, Lab_L2, Lab_a1, Lab_a2, Lab_b1, Lab_b2;

	// modeΪ��ɫ�ռ�ѡ�����ֵΪRGB�������RGB�ռ�ľ��룬��RGB�ռ���ɫ�
	if ("RGB" == mode)
	{
		// ��ͨ����ֱ�ӷ���������ĻҶȲ�ֵ
		if (1 == Img.channels())
		{
			double valueOfFirstPoint = Img.at<uchar>(point1.y, point1.x);
			cout << "firstPointColor:" << "(" << valueOfFirstPoint << ")";
			double valueOfSecondPoint = Img.at<uchar>(point2.y, point2.x);
			//            dDiffValue = sqrt(pow(valueOfFirstPoint,2)+pow(valueOfSecondPoint,2));
			dDiffValue = abs(valueOfSecondPoint - valueOfSecondPoint);
			cout << "secondPointColor:" << "(" << valueOfSecondPoint << ")";
			cout << dDiffValue;
		}
		if (3 == Img.channels())
		{
			b1 = Img.at<Vec3b>(point1.y, point1.x)[0];
			g1 = Img.at<Vec3b>(point1.y, point1.x)[1];
			r1 = Img.at<Vec3b>(point1.y, point1.x)[2];
			b2 = Img.at<Vec3b>(point2.y, point2.x)[0];
			g2 = Img.at<Vec3b>(point2.y, point2.x)[1];
			r2 = Img.at<Vec3b>(point2.y, point2.x)[2];
			double dAvg_r1r2 = (r1 + r2) / 2.0;
			cout << "RGB_colorSpace_" << "firstPointColor:(" << b1 << "," << g1 << "," << r1 << ")";
			cout << "RGB_ColorSpace_secondPointColor:(" << b2 << "," << g2 << "," << r2 << ")";
			//dDiffValue = sqrt(pow(abs(r2-r1),2)+pow(abs(g2-g1),2)+pow(abs(b2-b1),2));
			dDiffValue = 0.2 * sqrt((2 + dAvg_r1r2 / 256)*pow(abs(r2 - r1), 2) + 4 * pow(abs(g2 - g1), 2) + (2 + (255 - dAvg_r1r2) / 256)*pow(abs(b2 - b1), 2));
		}
	}
	else if ("Lab" == mode)
	{
		if (1 == Img.channels())
		{
			return false;
		}
		if (3 == Img.channels())
		{
			b1 = Img.at<Vec3b>(point1.y, point1.x)[0];
			g1 = Img.at<Vec3b>(point1.y, point1.x)[1];
			r1 = Img.at<Vec3b>(point1.y, point1.x)[2];
			b2 = Img.at<Vec3b>(point2.y, point2.x)[0];
			g2 = Img.at<Vec3b>(point2.y, point2.x)[1];
			r2 = Img.at<Vec3b>(point2.y, point2.x)[2];
			RGB2Lab(r1, g1, b1, Lab_L1, Lab_a1, Lab_b1);
			RGB2Lab(r2, g2, b2, Lab_L2, Lab_a2, Lab_b2);
			dDiffValue = sqrt(pow(abs(Lab_L2 - Lab_L1), 2) + pow(abs(Lab_a2 - Lab_a1), 2) + pow(abs(Lab_b2 - Lab_b1), 2));
		}
	}
	else
	{
		dDiffValue = 0.0;
	}

	return true;
}

bool CFunctionLib::SeparateTilesFromSrcImage(HObject ho_srcImage, HObject &ho_resultImage, map<string, int> KV_Map)
{
	// Local iconic variables
	HObject  ho_Region, ho_RegionOpening;
	HObject  ho_RegionDilation, ho_RegionClosing, ho_ImageReduced;
	HObject  ho_Edges, ho_SelectedContours, ho_UnionContours;
	HObject  ho_Rectangle, ho_ImageAffineTrans, ho_Region1, ho_RegionOpening1;
	HObject  ho_RegionDilation1, ho_RegionClosing1, ho_ImageReduced1;
	HObject  ho_Rectangle1, ho_ImageReduced2;

	// Local control variables
	HTuple  hv_Row, hv_Column, hv_Phi, hv_Length1;
	HTuple  hv_Length2, hv_PointOrder, hv_Area, hv_Row1, hv_Column1;
	HTuple  hv_HomMat2D, hv_Row11, hv_Column11, hv_Row2, hv_Column2;


	HTuple binaryThreshValue = 40; //(0 ~ 255)
	HTuple openingCircleValue = 20; //(0 ~ 1000)
	HTuple closingCircleValue = 50; //(0 ~ 1000)
	HTuple countorMinLimitValue = 1000; //(0 ~ 9999999)
	HTuple countorMaxLimitValue = 9999999;
	HTuple MaxDistAbs = 10; //Maximum distance of the contours' end points.


							// ��ͼƬ������ֵ��ֵ����
	Threshold(ho_srcImage, &ho_Region, binaryThreshValue, 255);

	// �����㣬��С������������
	OpeningCircle(ho_Region, &ho_RegionOpening, openingCircleValue);

	// ���Ͳ�����
	DilationCircle(ho_RegionOpening, &ho_RegionDilation, 10);

	// ���������������С����
	ClosingCircle(ho_RegionDilation, &ho_RegionClosing, closingCircleValue);

	// �ü�������
	ReduceDomain(ho_srcImage, ho_RegionClosing, &ho_ImageReduced);

	// ���������ر�Ե��ȡ��
	EdgesSubPix(ho_ImageReduced, &ho_Edges, "lanser2", 0.3, 10, 30);

	// ɸѡ������
	SelectContoursXld(ho_Edges, &ho_SelectedContours, "contour_length", countorMinLimitValue, countorMaxLimitValue, -0.5, 0.5);

	// ���ӶϿ���������
	UnionAdjacentContoursXld(ho_SelectedContours, &ho_UnionContours, 10, 1, "attr_keep");


	// ��Ͼ��α߽硣
	FitRectangle2ContourXld(ho_UnionContours, "tukey", -1, 0, 0, 3, 2, &hv_Row, &hv_Column, &hv_Phi, &hv_Length1, &hv_Length2, &hv_PointOrder);

	// ���ɾ��Ρ�
	GenRectangle2ContourXld(&ho_Rectangle, hv_Row, hv_Column, hv_Phi, hv_Length1, hv_Length2);

	// �����������������ĵ����ꡣ
	AreaCenter(ho_ImageReduced, &hv_Area, &hv_Row1, &hv_Column1);

	// �������任�ı任����
	VectorAngleToRigid(hv_Row1, hv_Column1, hv_Phi, hv_Row1, hv_Column1, HTuple(90).TupleRad(), &hv_HomMat2D);

	// ��Դͼ���з���任��
	AffineTransImage(ho_srcImage, &ho_ImageAffineTrans, hv_HomMat2D, "constant", "false");

	// ��������
	//CropDomain(ho_ImageReduced1, &ho_ImagePart);

	// ��ͼƬ��ֵ����ȡROI����
	Threshold(ho_ImageAffineTrans, &ho_Region1, binaryThreshValue, 255);

	// �����㣬��С������������
	OpeningCircle(ho_Region1, &ho_RegionOpening1, openingCircleValue);

	// ���Ͳ�����
	DilationCircle(ho_RegionOpening1, &ho_RegionDilation1, 10);

	// �����㣬����С����
	ClosingCircle(ho_RegionDilation1, &ho_RegionClosing1, 50);

	// �ü�ͼƬ
	ReduceDomain(ho_ImageAffineTrans, ho_RegionClosing1, &ho_ImageReduced1);

	//    // ���������ر�Ե��ȡ
	//    ThresholdSubPix(ho_ImageReduced1, &ho_Border, 100);

	//    // ������״����
	//    SelectShapeXld(ho_Border, &ho_SelectedXLD, "contlength", "or", 200, 9999999);

	//    // ������״�����������Ӧ������
	//    GenRegionContourXld(ho_SelectedXLD, &ho_Region2, "filled");

	// ��������ڽӾ���
	InnerRectangle1(ho_ImageReduced1, &hv_Row11, &hv_Column11, &hv_Row2, &hv_Column2);

	// ��������ڽӾ�������
	GenRectangle1(&ho_Rectangle1, hv_Row11, hv_Column11, hv_Row2, hv_Column2);

	// ��ȡĿ������
	ReduceDomain(ho_ImageAffineTrans, ho_Rectangle1, &ho_ImageReduced2);

	// ����Ŀ������
	CropDomain(ho_ImageReduced2, &ho_resultImage);

	return true;
}

bool CFunctionLib::ImageEqualDivided(Mat srcImage, vector<Mat> &vec_SubImages, int subImageNum)
{
	// ��ԴͼƬ�ȷ�Ϊ9��
	//subImageNum = 3;
	if (!srcImage.data)
	{
		return false;
	}

	// �����ͼƬ��
	vec_SubImages.clear();

	// ����Դͼ����ͼ�Ŀ�͸�
	int srcHeight, srcWidth, subHeight, subWidth;
	srcHeight = srcImage.rows;
	srcWidth = srcImage.cols;
	subHeight = srcHeight / subImageNum;
	subWidth = srcWidth / subImageNum;

	// ���зֿ������
	for (int j = 0; j < subImageNum; j++)
	{
		for (int i = 0; i < subImageNum; i++)
		{
			if (j < subImageNum - 1 && i < subImageNum - 1)
			{
				Mat temImage(subHeight, subWidth, CV_8UC3, Scalar(0, 0, 0));
				Mat imageROI = srcImage(Rect(i * subWidth, j * subHeight, temImage.cols, temImage.rows));
				addWeighted(temImage, 1.0, imageROI, 1.0, 0, temImage);
				vec_SubImages.push_back(temImage);
			}
			else
			{
				Mat temImage(srcHeight - (subImageNum - 1) * subHeight, srcWidth - (subImageNum - 1) * subWidth, CV_8UC3, Scalar(0, 0, 0));
				Mat imageROI = srcImage(Rect(i * subWidth, j * subHeight, temImage.cols, temImage.rows));
				addWeighted(temImage, 1.0, imageROI, 1.0, 0, temImage);
				vec_SubImages.push_back(temImage);
			}
		}
	}
}

bool CFunctionLib::MeanValueOfLab(Mat &srcImage, vector<float> &MeanValueOfLab_Lab)
{
	vector<float> Channel_L;
	vector<float> Channel_a;
	vector<float> Channel_b;

	uchar RGB_B = 0;
	uchar RGB_G = 0;
	uchar RGB_R = 0;

	float Lab_L = 0.0f;
	float Lab_a = 0.0f;
	float Lab_b = 0.0f;

	float MeanValueOfLab_L;
	float MeanValueOfLab_a;
	float MeanValueOfLab_b;

	MeanValueOfLab_Lab.clear();

	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			RGB_B = srcImage.at<Vec3b>(i, j)[0];
			RGB_G = srcImage.at<Vec3b>(i, j)[1];
			RGB_R = srcImage.at<Vec3b>(i, j)[2];
			RGB2Lab(RGB_R, RGB_G, RGB_B, Lab_L, Lab_a, Lab_b);
			Channel_L.push_back(Lab_L);
			Channel_a.push_back(Lab_a);
			Channel_b.push_back(Lab_b);
		}
	}

	float sum_L = accumulate(begin(Channel_L), end(Channel_L), 0.0);
	float sum_a = accumulate(begin(Channel_a), end(Channel_a), 0.0);
	float sum_b = accumulate(begin(Channel_b), end(Channel_b), 0.0);

	MeanValueOfLab_L = sum_L / Channel_L.size();
	MeanValueOfLab_a = sum_a / Channel_a.size();
	MeanValueOfLab_b = sum_b / Channel_b.size();

	MeanValueOfLab_Lab.push_back(MeanValueOfLab_L);
	MeanValueOfLab_Lab.push_back(MeanValueOfLab_a);
	MeanValueOfLab_Lab.push_back(MeanValueOfLab_b);

	return true;
}

bool CFunctionLib::MaxDistance(vector<vector<float> > vec_LabPoint, double &dMaxDistance)
{
	double temp = 0.0;
	for (int i = 0; i < vec_LabPoint.size(); i++)
	{
		for (int j = i + 1; j < vec_LabPoint.size(); j++)
		{
			temp = sqrt(pow((vec_LabPoint[i][0] - vec_LabPoint[j][0]), 2) + pow((vec_LabPoint[i][1] - vec_LabPoint[j][1]), 2) + pow((vec_LabPoint[i][2] - vec_LabPoint[j][2]), 2));
			dMaxDistance = dMaxDistance > temp ? dMaxDistance : temp;
		}
	}
	return true;
}

Mat CFunctionLib::HObject2Mat(HObject hoImage)
{
	HTuple hvChannel;
	HString cType;
	Mat image;
	ConvertImageType(hoImage, &hoImage, "byte");
	CountChannels(hoImage, &hvChannel);
	Hlong width = 0;
	Hlong height = 0;
	if (1 == hvChannel[0].I())
	{
		HImage hImage(hoImage);
		void *ptr = hImage.GetImagePointer1(&cType, &width, &height);
		int W = width;
		int H = height;
		image.create(H, W, CV_8UC1);
		unsigned char *pdata = static_cast<unsigned char *>(ptr);
		memcpy(image.data, pdata, W*H);
	}
	else if (3 == hvChannel[0].I()) {
		void *Rptr;
		void *Gptr;
		void *Bptr;
		HImage hImage(hoImage);
		hImage.GetImagePointer3(&Rptr, &Gptr, &Bptr, &cType, &width, &height);
		int W = width;
		int H = height;
		image.create(H, W, CV_8UC3);
		vector<cv::Mat> VecM(3);
		VecM[0].create(H, W, CV_8UC1);
		VecM[1].create(H, W, CV_8UC1);
		VecM[2].create(H, W, CV_8UC1);
		unsigned char *R = (unsigned char *)Rptr;
		unsigned char *G = (unsigned char *)Gptr;
		unsigned char *B = (unsigned char *)Bptr;
		memcpy(VecM[2].data, R, W*H);
		memcpy(VecM[1].data, G, W*H);
		memcpy(VecM[0].data, B, W*H);
		cv::merge(VecM, image);
	}
	return image;
}

bool CFunctionLib::CalculateMeanValueOfWholeTiles()
{
	Mat LabImage;
	Scalar Mean, std_Value;

	CFunctionLib *rgb2lab;
	rgb2lab = new CFunctionLib;
	vector<float> MeanValueOfLab_Lab;

	Mat srcImage = imread("E:\\hello.jpg");
	clock_t start, end;
	cvtColor(srcImage, LabImage, COLOR_BGR2Lab);

	meanStdDev(LabImage, Mean, std_Value);
	start = clock();
	rgb2lab->MeanValueOfLab(srcImage, MeanValueOfLab_Lab);
	end = clock();

	int time = end - start;

	printf("����ͼ��Lab�ռ���ͨ���ľ�ֵΪ��L:%.4f, a:%.4f, b:%.4f\n", MeanValueOfLab_Lab[0], MeanValueOfLab_Lab[1], MeanValueOfLab_Lab[2]);
	printf("����ʱ�䣺%d ms \n", time);
	cout << Mean << endl;

	return true;
}

bool CFunctionLib::MeanValueOfLab(Mat &srcImage, Scalar &mean, Scalar &std_value)
{
	Mat mask(srcImage.rows, srcImage.cols, CV_32FC3, cv::Scalar(0, 0, 0));

	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			RGB2Lab(srcImage.at<Vec3b>(i, j)[2], srcImage.at<Vec3b>(i, j)[1], srcImage.at<Vec3b>(i, j)[0], mask.at<Vec3f>(i, j)[0], mask.at<Vec3f>(i, j)[1], mask.at<Vec3f>(i, j)[2]);
		}
	}

	meanStdDev(mask, mean, std_value);

	return true;
}

//��ȡɫ�źϲ��������ȼ�
bool CFunctionLib::GetCombinedResultGrade(map<int, vector<int>> map_CombineColorModel, const int nOriGrade, int &nResultGrade)
{
	map<int, vector<int>>::iterator iter = map_CombineColorModel.begin();
	for ( ; iter != map_CombineColorModel.end(); iter++)
	{
		if (nOriGrade == iter->first)
		{
			nResultGrade = iter->first;
			break;
		}
		else
		{
			if (std::find(iter->second.begin(), iter->second.end(), nOriGrade) != iter->second.end())
			{
				nResultGrade = iter->first;
				break;
			}
		}
	}
	return true;
}

bool FunctionLib::ReadIniFile(const QString path)
{
	QSettings settings(path, QSettings::IniFormat);
	//settings.setIniCodec(QTextCodec::codecForName("system"));
	settings.setIniCodec("utf-8");
	string qstr = settings.value("MergeParam/MergeNum").toString().toStdString();
	
	printf("���");
	return true;
}

bool FunctionLib::WriteIniFile(const QString path)
{
	QSettings settings(path, QSettings::IniFormat);
	
	settings.setIniCodec("utf-8");
	//string a = "2021��5��21��17:19:07";
	QString b = "2021��5��21��17:55:53";
	settings.setValue("MergeParam/MergeNum", b);


	//cv::SimpleBlobDetector::compute()

	return false;
}

//sift����ƥ��Ͷ�λ��λ����׼
void FunctionLib::Sift_test(string imgPath1, string imgPath2)
{
	/*
	//�ο���https://mp.weixin.qq.com/s?__biz=Mzk0NjE2NDcxMw%3D%3D&chksm=c30b0502f47c8c148f32401288ebcdd133a1fdc3d9c8cbfaf707d96ef2caf0b782dfcfdb25fc&idx=1&mid=2247484292&scene=21&sn=be7716d03d3c15a2061629ebea801f44#wechat_redirect
	
	
	*/

	clock_t start, end;

	//imgPath1 = "E:/TestImage20210615/walk01.jpg";
	//imgPath2 = "E:/TestImage20210615/walk03.jpg";

	imgPath1 = "E:/hello.jpg";
	imgPath2 = "E:/hello3.jpg";
	/*Mat img1 = imread(imgPath1, IMREAD_GRAYSCALE);
	Mat img2 = imread(imgPath2, IMREAD_GRAYSCALE);*/

	Mat img1 = imread(imgPath1);
	Mat img2 = imread(imgPath2);

	imshow("img1", img1);
	imshow("img2", img2);

	start = clock();
	int kp_number{ 2000 };
	vector<KeyPoint> kp1, kp2;

	//���������������
	Ptr<SiftFeatureDetector> siftdtc = SiftFeatureDetector::create(kp_number);

	//���ͼ1�ļ�ֵ��
	siftdtc->detect(img1, kp1);
	//Mat outimg1;
	//drawKeypoints(img1, kp1, outimg1);
	//imshow("img1 keyPoints", outimg1);

	//���ͼ2�ļ�ֵ��
	siftdtc->detect(img2, kp2);
	//Mat outimg2;
	//drawKeypoints(img2, kp2, outimg2);
	//imshow("img2 keyPoints", outimg2);

	//SIFT ����������ȡ
	Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
	Mat descriptor1, descriptor2;
	extractor->compute(img1, kp1, descriptor1);
	extractor->compute(img2, kp2, descriptor2);

	//����������ƥ�����
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> matches;
	Mat img_matches;
	matcher->match(descriptor1, descriptor2,matches);

	//����ƥ�����о������;�����С
	double min_dist = matches[0].distance;
	double max_dist = matches[0].distance;

	for (int m = 0; m < matches.size(); m++)
	{
		if (matches[m].distance < min_dist)
		{
			min_dist = matches[m].distance;
		}
		if (matches[m].distance > max_dist)
		{
			max_dist = matches[m].distance;
		}
	}
	cout << "min dist=" << min_dist << endl;
	cout << "max dist=" << max_dist << endl;

	vector<DMatch> goodMatches;
	for (int i = 0; i < matches.size(); i++)   //ɸѡ���Ϻõ�ƥ����
	{
		if (matches[i].distance < 0.25*max_dist)  //С��������һ�������ĵ������Ϊ�ϸ񣬷����޳�
		{
			goodMatches.push_back(matches[i]);
		}
	}

	//cout << "The number of good matches:" << goodMatches.size() << endl;

	//Mat good_img_matches;
	//drawMatches(img1, kp1, img2, kp2, goodMatches, good_img_matches);
	//imshow("good_img_matches", good_img_matches);


	//�����ϳ����޳�ʱ�����������������ɸѡ����
	vector <KeyPoint> good_kp1, good_kp2;
	for (int i = 0; i < goodMatches.size(); i++)
	{
		good_kp1.push_back(kp1[goodMatches[i].queryIdx]);
		good_kp2.push_back(kp2[goodMatches[i].trainIdx]);
	}



	//��ȡƥ��������Ե�����ֵ
	vector <Point2f> p01, p02;
	for (int i = 0; i < goodMatches.size(); i++)
	{
		p01.push_back(good_kp1[i].pt);
		p02.push_back(good_kp2[i].pt);
	}

	//RANSAC�㷨��һ���޳���ƥ����
	vector<uchar> RANSACStatus;//���Ա��ÿһ��ƥ����״̬������0��Ϊ��㣬����1��Ϊ�ڵ㡣
	findFundamentalMat(p01, p02, RANSACStatus, FM_RANSAC, 3);//p1 p2����Ϊfloat��
	vector<Point2f> f1_features_ok;
	vector<Point2f> f2_features_ok;
	for (int i = 0; i < p01.size(); i++)   //�޳�����ʧ�ܵ�
	{
		if (RANSACStatus[i])
		{
			f1_features_ok.push_back(p01[i]);       //ͼ1�ĵ�
			f2_features_ok.push_back(p02[i]);     //ͼ2�Ķ�Ӧ��
		}
	}

	//�������任����
	Mat T = cv::estimateRigidTransform(f2_features_ok, f1_features_ok, true);   //false��ʾ���Ա任

	Mat affine_out;
	//�����ز���
	warpAffine(img2, affine_out, T, img2.size(), INTER_CUBIC);
	end = clock();
	int cost = end - start;

	cout << "sift_test time_cost: " << cost << " ms." << endl;
	//namedWindow("affine_result", cv::WINDOW_NORMAL);
	imshow("affine_result", affine_out);
	imwrite("E:/TestImage20210615/sift_test.jpg", affine_out);


	waitKey(0);
}

//�ߵ���
void FunctionLib::BlobDectector(string imgPath)
{
	imgPath = "E:/TestImage20210615/simpleBlobdetector_1.jpg";
	Mat image = imread(imgPath);

	clock_t start, end;
	start = clock();
	vector<KeyPoint> keyPoints;
	SimpleBlobDetector::Params params;
	
	//��ֵ����
	params.minThreshold = 0;
	params.maxThreshold = 255;
	params.thresholdStep = 5;

	params.filterByColor = true;
	params.blobColor = 0;
	//�������
	params.filterByArea = true;
	params.minArea = 10;
	params.maxArea = 500;

	//͹��״
	params.filterByCircularity = true;
	params.minCircularity = 0.8f;
	params.maxCircularity = 0.999999;

	//����״
	params.filterByConvexity = false;
	params.minConvexity = 0.2f;
	params.maxConvexity = 0.999999;

	//Բ�ο���
	params.filterByInertia = true;
	params.minInertiaRatio = 0.1f;
	params.maxInertiaRatio = 0.999999;


	
	Ptr<SimpleBlobDetector> blobDetect = SimpleBlobDetector::create(params);
	
	blobDetect->detect(image, keyPoints);
	cout << keyPoints.size() << endl;
	end = clock();
	int cost = end - start;

	cout << "SimpleBlobDetector_test time_cost: " << cost << " ms." << endl;
	drawKeypoints(image, keyPoints, image, Scalar(0, 255, 0));

	namedWindow("blobs",WINDOW_NORMAL);
	imshow("blobs", image);
	imwrite("E:/TestImage20210615/simpleBlobdetector_1_result.jpg", image);
	waitKey();

}

//ë�̼��
void FunctionLib::GlitchDetector(string imgPath)
{
	imgPath = "E:/TestImage20210615/fin1.png";
	//����ͼ��
	Mat srcImg = imread(imgPath, 1);
	clock_t start, end;
	start = clock();
	Mat GrayImg;
	cvtColor(srcImg, GrayImg, COLOR_BGR2GRAY);
	//��ֵ��ͼ��
	Mat binaryImg;
	threshold(GrayImg, binaryImg, 100, 255, THRESH_BINARY);
	//imshow("binaryImg", binaryImg);

	//�Զ����
	Mat element = getStructuringElement(MORPH_RECT, Size(100, 100));

	//������
	Mat close_result;
	morphologyEx(binaryImg, close_result, MORPH_CLOSE, element);
	//imshow("close", close_result);
	Mat imgDiff;
	absdiff(binaryImg, close_result, imgDiff);
	//imshow("diff", imgDiff);

	/*Mat diff_Result;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(30, 30));
	morphologyEx(imgDiff, diff_Result, MORPH_CLOSE, kernel);
	imshow("diff_Result", diff_Result);*/

	//�Զ����
	Mat eroImg;
	Mat eroElement = getStructuringElement(MORPH_RECT, Size(5, 5));
	erode(imgDiff, eroImg, eroElement);
	Mat dist;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(eroImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	
	for (int i = 0; i<contours.size(); i++)
	{
		////contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���
		//for (int j = 0; j<contours[i].size(); j++)
		//{
		//	//���Ƴ�contours���������е����ص�
		//	Point P = Point(contours[i][j].x, contours[i][j].y);
		//	contours.at<uchar>(P) = 255;
		//}

		////���hierarchy��������
		//char ch[256];
		//sprintf(ch, "%d", i);
		//string str = ch;
		//cout << "����hierarchy�ĵ�" << str << " ��Ԫ������Ϊ��" << endl << hierarchy[i] << endl << endl;

		//��������
		drawContours(srcImg, contours, i, Scalar(0,0,255), 3, 8, hierarchy);
	}
	end = clock();
	int time_cost = end - start;

	cout << "GlitchDetector time_cost: " << time_cost << "ms." << endl;
	imshow("fin1_result", srcImg);
	imwrite("E:/TestImage20210615/fin1_result.png",srcImg);

	waitKey(0);

}

//���ۼ��
void FunctionLib::surface_scratch(string imgPath)
{
	Mat image, imageMean, imageDiff, Mask;
	image = imread("E:/TestImage20210615/surface_scratch.png");


	clock_t start, end;
	int cost = 0;

	start = clock();
	//��ֵģ��
	blur(image, imageMean, Size(13, 13));
	//imshow("imageMean", imageMean);
	//waitKey(0);

	//ͼ����
	subtract(imageMean, image, imageDiff);
	//imshow("imageDiff", imageDiff);
	//waitKey(0);

	//ͬ��̬��ֵ�ָ�dyn_threshold
	threshold(imageDiff, Mask, 5, 255, THRESH_BINARY);
	//imshow("Mask", Mask);
	//waitKey(0);

	Mat imageGray;
	cvtColor(Mask, imageGray, COLOR_BGR2GRAY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(Mask.size(), CV_8U);
	int j = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		Moments moms = moments(Mat(contours[i]));
		
		//��׾ؼ�Ϊ��ֵͼ��������m00Ϊ���������m10Ϊ�������ġ�
		double area = moms.m00;

		if (area > 20 && area < 1000)
		{
			drawContours(drawing, contours, i, Scalar(255), FILLED, 8, hierarchy, 0, Point());
			j = j + 1;
		}
	}

	Mat element15(3, 3, CV_8U, Scalar::all(1));
	Mat close;
	morphologyEx(drawing, close, MORPH_CLOSE, element15);
	//imshow("drawing", drawing);
	
	vector<vector<Point>> contours1;
	vector<Vec4i> hierarchy1;
	findContours(close, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//imshow("close", close);
	//imwrite("E:/TestImage20210615/surface_scratch_close.png", close);
	j = 0;
	int m = 0;
	for (int i = 0; i < contours1.size(); i++)
	{
		Moments moms = moments(Mat(contours1[i]));
		double area = moms.m00;

		double area1 = contourArea(contours1[i]);
		if (area > 50 && area < 100000)
		{
			drawContours(image, contours1, i, Scalar(0, 0, 255), FILLED, 8, hierarchy1, 0, Point());
			j = j + 1;
		}
		else if (area >= 0 && area <= 50)
		{
			drawContours(image, contours1, i, Scalar(255, 0, 0), FILLED, 8, hierarchy1, 0, Point());
			m = m + 1;
		}
	}
	end = clock();
	cost = end - start;
	cout << "surface_scratch cost time: " << cost << " ms." << endl;
	char t[256];
	sprintf_s(t, "%01d", j);
	string s = t;
	string txt = "LONG NG: " + s;
	putText(image, txt, Point(20, 30), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 8);

	sprintf_s(t, "%01d", m);
	txt = "Short NG: " + s;
	putText(image, txt, Point(20, 60), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 2, 8);
	imshow("output", image);
	imwrite("E:/TestImage20210615/surface_scratch_output.png",image);
	waitKey(0);
}

//�ǵ���
void FunctionLib::cornerHarris(string imgPath)
{
	imgPath = "E:/TestImage20210615/CornerHarrisDetector_1.jpg";
	clock_t start, end;
	Mat srcImg = imread(imgPath);
	start = clock();
	Mat src_gray;
	cvtColor(srcImg, src_gray, COLOR_RGB2GRAY);
	Mat cornerStrength;
	cornerHarris(src_gray, cornerStrength, 2, 3, 0.04);
	threshold(cornerStrength, cornerStrength, 0.0001, 255, THRESH_BINARY);
	//vector<vector<Point>> contours;
	//vector<Vec4i> hierarchy;

	//findContours(cornerStrength, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

	//for (int i = 0; i<contours.size(); i++)
	//{
	//	//��������
	//	drawContours(srcImg, contours, i, Scalar(0, 0, 255), 3, 8, hierarchy);
	//}

	/// Normalizing
	Mat  dst_norm, dst_norm_scaled;
	normalize(cornerStrength, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > 100)
			{
				circle(srcImg, Point(i, j), 5, Scalar(0,0,255), 1, 8, 0);
			}
		}
	}
	end = clock();
	int time_cost = end - start;
	cout << "CornerHarrisDetector time_cost: " << time_cost << " ms." << endl;
	imshow("shiyan", srcImg);
	imwrite("E:/TestImage20210615/CornerHarrisDetector_1_result.jpg",srcImg);
	waitKey(0);
}


//Բ�����
bool FunctionLib::fitCircle(string imgPath)
{
	imgPath = "E:/TestImage20210615/20210702_FitCircle_1.jpg";
	Mat srcImage = imread(imgPath);
	Mat grayImg;
	cvtColor(srcImage, grayImg, COLOR_BGR2GRAY);
	blur(grayImg, grayImg, Size(3,3));
	
	Mat binaryImg;
	threshold(grayImg, binaryImg, 0, 255, THRESH_OTSU);
	//imwrite("E:/1-work/code/code_python/opencvTest/image/FitCircle_1_20210702.jpg", binaryImg);
	//imshow("binary", binaryImg);
	
	binaryImg = binaryImg > 200;
	vector<vector<Point> > contours;
	vector<Vec4i>hierarchy;
	Mat dst = Mat::zeros(binaryImg.rows, binaryImg.cols, CV_8UC3);
	//cout << binaryImg << endl;
	findContours(binaryImg, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	
	//������
	if (!contours.empty() && !hierarchy.empty())
	{
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, contours, idx, color, 1, 8, hierarchy);
		}
	}
	



	namedWindow("Connected components", 1);
	imshow("Connected components", dst);

	waitKey(0);
	return true;
}

//�������
bool FunctionLib::fitCurve(string imgPath)
{
	//�������ڻ��Ƶ�����ɫ����ͼ��
	cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
	image.setTo(cv::Scalar(100, 0, 0));

	//������ϵ�  
	/*std::vector<cv::Point> points;
	points.push_back(cv::Point(100., 58.));
	points.push_back(cv::Point(150., 70.));
	points.push_back(cv::Point(200., 90.));
	points.push_back(cv::Point(252., 140.));
	points.push_back(cv::Point(300., 220.));
	points.push_back(cv::Point(350., 400.));*/

	std::vector<cv::Point> points;
	points.push_back(cv::Point(100., 50.));
	points.push_back(cv::Point(150., 100.));
	points.push_back(cv::Point(200., 50.));
	points.push_back(cv::Point(250., 100.));
	points.push_back(cv::Point(300., 50.));
	points.push_back(cv::Point(350., 100.));

	//����ϵ���Ƶ��հ�ͼ��  
	for (int i = 0; i < points.size(); i++)
	{
		cv::circle(image, points[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	//��������
	//cv::polylines(image, points, false, cv::Scalar(0, 255, 0), 1, 8, 0);

	cv::Mat A;

	polynomial_curve_fit(points, 5, A);
	std::cout << "A = " << A << std::endl;

	std::vector<cv::Point> points_fitted;

	for (int x = 0; x < 400; x++)
	{
		double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
			A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);

		points_fitted.push_back(cv::Point(x, y));
	}
	cv::polylines(image, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);

	cv::imshow("image", image);

	cv::waitKey(0);
	return 0;

}

bool FunctionLib::polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//�������X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//�������Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//������A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}

//ֱ�����
bool FunctionLib::fitLine(string imgPath)
{
	Mat sobel_x, sobel_y, sobelxy,sobelx,sobely;
	imgPath = "E:/TestImage20210615/20210625_3_fitLine.jpg";
	Mat srcImg = imread(imgPath,0);
	Sobel(srcImg, sobel_x, CV_64F, 1, 0, 3);
	//convertScaleAbs(sobel_x, sobelx);


	Sobel(srcImg, sobel_y, CV_64F, 0, 1, 3);
	//convertScaleAbs(sobel_y,  sobely);

	addWeighted(sobel_x, 0.5, sobel_y, 0.5,1, sobelxy);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat binaryImg;
	threshold(sobelxy, binaryImg, 0, 255, THRESH_BINARY);
	//cout << binaryImg << endl;
	imshow("binaryImg", binaryImg);
	waitKey(0);
	findContours(binaryImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/*
	int j = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		Moments moms = moments(Mat(contours[i]));

		//��׾ؼ�Ϊ��ֵͼ��������m00Ϊ���������m10Ϊ�������ġ�
		double area = moms.m00;

		if (area > 20 && area < 1000)
		{
			drawContours(srcImg, contours, i, Scalar(255), FILLED, 8, hierarchy, 0, Point());
			j = j + 1;
		}
	}
	*/

	cv::Vec4f line_para;
	cv::fitLine(contours, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);


	//��ȡ��бʽ�ĵ��б��
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];

	double k = line_para[1] / line_para[0];

	//����ֱ�ߵĶ˵�(y = k(x - x0) + y0)
	cv::Point point1, point2;
	point1.x = 0;
	point1.y = k * (0 - point0.x) + point0.y;
	point2.x = 2 * point0.x;
	point2.y = k * (point2.x - point0.x) + point0.y;

	cv::line(srcImg, point1, point2, cv::Scalar(0, 0, 255), 2, 8, 0);

	cv::imshow("image", srcImg);
	cv::waitKey(0);
	return true;
}

//ֱ�߼��
bool FunctionLib::lineDetector(string imgPath, vector<Vec4f> &lines)
{
	imgPath = "E:/TestImage20210615/20210707_1_LineCrossPoint_1.jpg";
	Mat srcImg = imread(imgPath);
	namedWindow("srcImage", WINDOW_NORMAL);
	imshow("srcImage", srcImg);
	clock_t start, end;
	start = clock();
	Mat grayImg;
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	Mat edges;
	Canny(grayImg, edges, 50, 150, 3);
	//vector<Vec4f> lines;
	HoughLinesP(edges, lines, 1, PI / 180.0, 200,0.0,200);
	for (int i = 0; i < lines.size(); i++)
	{
		cout << lines[i] << endl;
		Vec4f hLine = lines[i];
		line(srcImg, Point(hLine[0], hLine[1]), Point(hLine[2], hLine[3]), Scalar(0, 0, 255), 3,LINE_AA);
	}
	end = clock();

	int time_cost = end - start;
	cout << "houghLinesP Detect Lines time_cost: " << time_cost << " ms." << endl;
	namedWindow("srcImg_result", WINDOW_NORMAL);
	
	vector<Point2f> OutputPointList;
	clock_t GetCrossPoint_start, GetCrossPoint_end;
	GetCrossPoint_start = clock();
	GetMultyLineCrossPoint(lines, OutputPointList);
	GetCrossPoint_end = clock();
	int GetCrossPoint_timeCost = GetCrossPoint_end - GetCrossPoint_start;
	//cout << "GetMultyLineCrossPoint time_cost: " << GetCrossPoint_timeCost << "ms." << endl;
	DrawPoint(OutputPointList, srcImg);
	double Distance;
	clock_t distance_start, distance_end;
	distance_start = clock();
	Distance_PointToLine(lines[0], Point(152, 90),Distance);
	distance_end = clock();
	int distance_timeCost = distance_end - distance_start;
	cout << "distance calculate time_cost: " << distance_timeCost << "ms." << endl;
	cout << "Point(152, 90) to line Distance: "<<Distance <<" pixel."<< endl;
	imwrite("E:/TestImage20210615/20210707_1_LineCrossPoint_1_result.jpg", srcImg);
	imshow("srcImg_result", srcImg);
	waitKey(0);
	return true;
}

//��������ֱ�߽���
Point2f FunctionLib::GetLineCrossPoint(Vec4i LineA, Vec4i LineB)
{
	Point2f crossPoint;
	double ka, kb;
	if ((LineA[2] == LineA[0])&&(LineB[2] != LineB[0]))
	{
		kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); //���LineBб��
		crossPoint.x = LineA[2];
		crossPoint.y = kb*(crossPoint.x - LineB[0]) + LineB[1];
	}
	else if ((LineA[2] != LineA[0]) && (LineB[2] == LineB[0]))
	{
		ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); //���LineAб��
		crossPoint.x = LineB[2];
		crossPoint.y = ka*(crossPoint.x - LineA[0]) + LineA[1];
	}
	else if ((LineA[2] == LineA[0]) && (LineB[2] == LineB[0]))
	{
		crossPoint.x = -1;
		crossPoint.y = -1;
	}
	else
	{
		ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); //���LineAб��
		kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); //���LineBб��


		if (ka != kb)
		{
			crossPoint.x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb);
			crossPoint.y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb);
		}
		else
		{
			crossPoint.x = -1;
			crossPoint.y = -1;
		}
	}

	return crossPoint;
}

//�������ֱ������֮��Ľ���
bool FunctionLib::GetMultyLineCrossPoint(vector<Vec4f> InputLines,vector<Point2f> &OutputPointList)
{
	OutputPointList.clear();
	clock_t start, end;
	start = clock();
	for (int i = 0; i < InputLines.size(); i++)
	{
		for (int j = i+1; j < InputLines.size(); j++)
		{
			Point2f CrossPoint = GetLineCrossPoint(InputLines[i], InputLines[j]);
			OutputPointList.push_back(CrossPoint);
		}
	}
	end = clock();
	int time_cost = end - start;
	cout << "GetMultyLineCrossPoint time_cost: " << time_cost << " ms." << endl;
 	return true;
}

//��ͼƬ�ϻ���
bool FunctionLib::DrawPoint(vector<Point2f> InputPointList, Mat &DrawImage)
{
	for (int i = 0; i < InputPointList.size(); i++)
	{
		if (InputPointList[i].x >= 0 && InputPointList[i].x <= DrawImage.cols && InputPointList[i].y >= 0 && InputPointList[i].y <= DrawImage.rows)
		{
			circle(DrawImage, InputPointList[i], 6, Scalar(0, 255, 0), 2, LINE_AA);
		}
	}
	return true;
}

//����㵽ֱ�ߵľ���
bool FunctionLib::Distance_PointToLine(Vec4i Line, Point P, double &Distance)
{
	Point P1, P2;
	P1.x = Line[0];
	P1.y = Line[1];
	P2.x = Line[2];
	P2.y = Line[3];

	//ֱ�߷���һ��ʽΪ��Ax + By + C = 0; KΪ��бʽ���̵�б�ʡ�
	double k, b, A, B, C;
	if (P1.x == P2.x)
	{
		Distance = abs(P.x - P1.x);
	}
	else if(P1.y == P2.y)
	{
		Distance = abs(P.y - P1.y);
	}
	else 
	{
		// y = kx + b,���У�kx - y + b = 0,ֱ�ߵ�һ��ʽ����
		k = (double)(P2.y - P1.y) / (double)(P2.x - P1.x); //���LineBб�ʡ�
		b = P1.y - k * P1.x; // ���ֱ�ߵĽؾ࣬

		A = k;
		B = -1;
		C = b;
		Distance = abs(A * P.x + B * P.y + C) / sqrt(pow(A, 2) + pow(B,2)); //�㵽ֱ�ߵľ��빫ʽ
	}

	return true;
}

//������תһ���Ƕ�
bool FunctionLib::GetRotatePoints(vector<Point> InputPoints, vector<Point2d> &OutputPoints, Point InputCenterPoint, double InputDoubleAngle, bool InputBoolDirection)
{
	/**********************************************************************
	*	��֪��P0(x0,y0)�����ĵ�Pc(xc,yc),�Ƕ�angle = ������ת֮��ĵ������P'(x',y')Ϊ��
	*
	*	x' = (x0 - xc) * cos�� - (y0 - yc) * sin�� + xc
	*   y' = (y0 - yc) * cos�� - (x0 - xc) * sin�� + yc
	*
	*/

	if (InputPoints.size() >= 2)
	{
		for (int i = 0; i < InputPoints.size(); i++)
		{
			Point2d tmp_Point;
			tmp_Point.x = (InputPoints[i].x - InputCenterPoint.x) * cos(InputDoubleAngle) - (InputPoints[i].y - InputCenterPoint.y) * sin(InputDoubleAngle) + InputCenterPoint.x;
			tmp_Point.y = (InputPoints[i].y - InputCenterPoint.y) * cos(InputDoubleAngle) + (InputPoints[i].x - InputCenterPoint.x) * sin(InputDoubleAngle) + InputCenterPoint.y;
			OutputPoints.push_back(tmp_Point);
		}
	}

	return true;
}


bool FunctionLib::RotateTest()
{
	vector<Point> vec_Point;
	Point p1 = Point(100, 100);
	Point p2 = Point(100, 300);
	Point p3 = Point(300, 300);
	Point p4 = Point(300, 100);

	vec_Point.push_back(p1);
	vec_Point.push_back(p2);
	vec_Point.push_back(p3);
	vec_Point.push_back(p4);

	Mat img = Mat::zeros(1000, 1000, CV_8UC3);

	vector<Point2d> outputPoint;
	GetRotatePoints(vec_Point, outputPoint, Point(500, 500), -PI / 2, 0);
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
	return true;
}

void FunctionLib::rotatePoint(double angle, Point &rotate_pt, Point origin_pt, Point center_pt)
{
	double x0 = center_pt.x;
	double y0 = center_pt.y;

	double x = origin_pt.x;
	double y = origin_pt.y;

	rotate_pt.x = (x - x0) * cos(angle* PI / 180) - (y - y0) * sin(angle* PI / 180) + x0;
	rotate_pt.y = (x - x0) * sin(angle* PI / 180) + (y - y0) * cos(angle* PI / 180) + y0;
}


/*
void FunctionLib::on_CornerHarris(int thresh, void *,Mat g_srcImage,Mat g_srcImage1)
{
	Mat dstImage;//Ŀ��ͼ
	Mat normImage;//��һ�����ͼ
	Mat scaledImage;//���Ա任��İ�λ�޷�������ͼ��

	//���㵱ǰ��Ҫ��ʾ������ͼ���������һ�ε��ôκ���ʱ���ǵ�ֵ��
	dstImage = Mat::zeros(g_srcImage.size(), CV_32FC1);
	g_srcImage1 = g_srcImage.clone();

	//���нǵ���
	//������������ʾ�����С�����ĸ�������ʾSobel���ӿ׾���С�������������ʾHarris����
	cornerHarris(g_srcImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);

	//��һ����ת��
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);//����һ�����ͼ���Ա任��8λ�޷�������

	//����⵽�ģ��ҷ�����ֵ�����Ľǵ���Ƴ���
	for (int j = 0; j < normImage.rows; j++)
	{
		for (int i = 0; i < normImage.cols; i++)
		{
			//Mat::at<float>(j,i)��ȡ����ֵ��������ֵ�Ƚ�
			if ((int)normImage.at<float>(j, i) > thresh + 80)
			{
				circle(g_srcImage1, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}
	imshow("�ǵ���", g_srcImage1);
	imshow("�ǵ���2", scaledImage);

}
*/