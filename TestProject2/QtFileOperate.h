
#if !defined(QTFILEOPERATE_H)
#define QTFILEOPERATE_H

#include <QtCore\QString>
#include <QtCore\QMutex>
#define WIDTH_BYTES(bits) (((bits) + 31) / 32 * 4)
#pragma execution_character_set("utf-8")

class CQtFileOperate
{

public:
    CQtFileOperate();
    ~CQtFileOperate();

    QString GetLastError();

	// 获取本执行程序所在的路径
    QString GetCurrentAppPath();


	// 判断路径是否存在
    bool IsPathExist(QString strPath);

	//创建一个多层目录，如果存在就不创建
    bool CreateMultiLevelPath(QString strPath);

	
	//删除当前目录下所有文件及文件夹
    bool DeleteDirectory(QString strPath);


    //拷贝文件夹：
    //如果strToPath不存在：自动生成strToPath，并 将strFromPath中的内容（不包含strFromPath），拷贝到strToPath中
    //如果strToPath存在， 返回 false
    bool CopyFolder(QString strFromPath, QString strToPath);


    //重命名 F:\\Bin\\Model","F:\\Bin\\Model123" 表示将Model重命名为Model123
    bool ReNameFolder(QString strFromPath, QString  strToPath);


	//浏览对话框,strInitPath限制路径   返回的文件夹全路径
    bool BrowseFolder(QString strInitPath, QString &strBrownPath);

    //删除某文件夹下，年月及之前的文件夹，例如：F:\Bin\Model\log    子文件夹格式2018年08月
    bool DeleteFolderByYearMonDay(QString  strFolder, int iYear, int iMonth, int iDay);
    ///////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////


	//判断文件是否存在
    bool IsFileExist(QString strFileFullPath);
    bool IsFileExist(std::string strFileFullPath);


    bool FileDelete(QString strFileFullPath);


	//strExt = exe或者bmp  或者*
    bool FileBrowse(QString strInitPath, QString strExt, QString &strFileFullPath);


	//strExt = exe或者bmp  或者*
    bool FileSave(QString strInitPath, QString strExt, QString &strFilePath);

    ///////////////////////////

    bool MyWritePrivateProfileString(QString strApp, QString strKey, QString strContent,QString strFilePath);
    QString MyReadPrivateProfileString(QString strApp, QString strKey, QString strDefault, QString strFilePath);

    bool MyWritePrivateProfileInt(QString strApp, QString strKey, int iContent, QString strFilePath);
    int MyReadPrivateProfileInt(QString strApp, QString strKey, int iDefault, QString strFilePath);


    bool MyWritePrivateProfileDouble(QString strApp, QString strKey,  double dbContent, QString strFilePath);
    double MyReadPrivateProfileDouble(QString strApp, QString strKey, double dbDefault, QString strFilePath);

    bool MyWritePrivateProfileBool(QString strApp, QString strKey, bool bContent, QString strFilePath);
    bool MyReadPrivateProfileBool(QString strApp, QString strKey, bool bDefault, QString strFilePath);

    //
    bool MyWritePrivateProfileString(std::string strApp, std::string strKey, std::string strContent,std::string strFilePath);
    std::string MyReadPrivateProfileString(std::string strApp, std::string strKey, std::string strDefault, std::string strFilePath);

    bool MyWritePrivateProfileInt(std::string strApp, std::string strKey, int iContent, std::string strFilePath);
    int MyReadPrivateProfileInt(std::string strApp, std::string strKey, int iDefault, std::string strFilePath);


    bool MyWritePrivateProfileDouble(std::string strApp, std::string strKey,  double dbContent, std::string strFilePath);
    double MyReadPrivateProfileDouble(std::string strApp, std::string strKey, double dbDefault, std::string strFilePath);

    bool MyWritePrivateProfileBool(std::string strApp, std::string strKey, bool bContent, std::string strFilePath);
    bool MyReadPrivateProfileBool(std::string strApp, std::string strKey, bool bDefault, std::string strFilePath);

   //////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
    //写日志

    bool WriteLog(QString strLogFilePath, QString strContent);
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//获得内存剩余容量函数，单位：M
     long GetSurplusMemorySize();

	//获得当前硬盘分区的剩余容量，（返回-1表示失败单位：M） （输入 D:   E:）
    long GetSurplusCurrentPartitionSize(QString strDrivePath);


	//杀死当前进程
	void ShutCurrentProgcess(); 

	//////////////////////////////////////////////////////////////////////////


	//写调试信息
	//用法 PrintfDebugLog("[PCB]%d starts to run\n", 1);
    void PrintfDebugLog(const char * strOutputString, ...);
    void PrintfDebugLog(QString strOutputString);


	//////////////////////////////////////////////////////////////////////////
	//要求nTextLen = nOutLen；
    bool Encrypt(const char szText[], unsigned int nTextLen, char szOutString[], unsigned int nOutLen);
	bool Decrypt(const char szText[], unsigned int nTextLen, char szOutString[], unsigned int nOutLen);

    QString Encrypt(const QString &strInput);
    QString Decrypt(const QString &strInput);
	//////////////////////////////////////////////////////////////////////////
	//与图形图像文件相关的
    bool SaveDIBImageToBMPFile(QString strFileName, const char* pImageBuff, long lImageWidth, long lImageHeight, long lImageBitCount);
    bool CutDIBImage(char *pDest, long xDest, long yDest, long lDestWidth, long lDestHeight, const char* pSrcImageBuff, long lSrcImageWidth, long lSrcImageHeight, long lSrcImageBitCount);
	bool CutPlaneRGBImage(char *pDest, long xDest, long yDest, long lDestWidth, long lDestHeight, const char* pSrcImageBuff, long lSrcImageWidth, long lSrcImageHeight);
	bool ConvertRGBToPlaneR_G_B(char *pDestImageBuffR, char *pDestImageBuffG, char *pDestImageBuffB,
		const char* pSrcImageBuff, long lSrcImageWidth, long lSrcImageHeight);
	bool ConvertPlaneR_G_BToRGB(char* pDestImageBuff, const char *pSrcImageBuffR, const char *pSrcImageBuffG, const char *pSrcImageBuffB,
		long lSrcImageWidth, long lSrcImageHeight);
	bool  ConvertRGBToPlaneRGB(char *pDestPlaneRGBBuff, const char* pSrcImageRGBBuff, long lSrcImageWidth, long lSrcImageHeight);
	bool  ConvertPlaneRGBToRGB(char* pDestImageRGBBuff, const char *pSrcPlaneRGBBuff, long lSrcImageWidth, long lSrcImageHeight);
	



private:
     QString m_strErrorInfo;
	 QMutex m_mutexBuffer;
};



#endif // !define(FileOperate_h_)
