
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

	// ��ȡ��ִ�г������ڵ�·��
    QString GetCurrentAppPath();


	// �ж�·���Ƿ����
    bool IsPathExist(QString strPath);

	//����һ�����Ŀ¼��������ھͲ�����
    bool CreateMultiLevelPath(QString strPath);

	
	//ɾ����ǰĿ¼�������ļ����ļ���
    bool DeleteDirectory(QString strPath);


    //�����ļ��У�
    //���strToPath�����ڣ��Զ�����strToPath���� ��strFromPath�е����ݣ�������strFromPath����������strToPath��
    //���strToPath���ڣ� ���� false
    bool CopyFolder(QString strFromPath, QString strToPath);


    //������ F:\\Bin\\Model","F:\\Bin\\Model123" ��ʾ��Model������ΪModel123
    bool ReNameFolder(QString strFromPath, QString  strToPath);


	//����Ի���,strInitPath����·��   ���ص��ļ���ȫ·��
    bool BrowseFolder(QString strInitPath, QString &strBrownPath);

    //ɾ��ĳ�ļ����£����¼�֮ǰ���ļ��У����磺F:\Bin\Model\log    ���ļ��и�ʽ2018��08��
    bool DeleteFolderByYearMonDay(QString  strFolder, int iYear, int iMonth, int iDay);
    ///////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////


	//�ж��ļ��Ƿ����
    bool IsFileExist(QString strFileFullPath);
    bool IsFileExist(std::string strFileFullPath);


    bool FileDelete(QString strFileFullPath);


	//strExt = exe����bmp  ����*
    bool FileBrowse(QString strInitPath, QString strExt, QString &strFileFullPath);


	//strExt = exe����bmp  ����*
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
    //д��־

    bool WriteLog(QString strLogFilePath, QString strContent);
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//����ڴ�ʣ��������������λ��M
     long GetSurplusMemorySize();

	//��õ�ǰӲ�̷�����ʣ��������������-1��ʾʧ�ܵ�λ��M�� ������ D:   E:��
    long GetSurplusCurrentPartitionSize(QString strDrivePath);


	//ɱ����ǰ����
	void ShutCurrentProgcess(); 

	//////////////////////////////////////////////////////////////////////////


	//д������Ϣ
	//�÷� PrintfDebugLog("[PCB]%d starts to run\n", 1);
    void PrintfDebugLog(const char * strOutputString, ...);
    void PrintfDebugLog(QString strOutputString);


	//////////////////////////////////////////////////////////////////////////
	//Ҫ��nTextLen = nOutLen��
    bool Encrypt(const char szText[], unsigned int nTextLen, char szOutString[], unsigned int nOutLen);
	bool Decrypt(const char szText[], unsigned int nTextLen, char szOutString[], unsigned int nOutLen);

    QString Encrypt(const QString &strInput);
    QString Decrypt(const QString &strInput);
	//////////////////////////////////////////////////////////////////////////
	//��ͼ��ͼ���ļ���ص�
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
