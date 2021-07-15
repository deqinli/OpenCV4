#include "QtFileOperate.h"
#include <QtCore\qcoreapplication.h>
//#include <QCoreApplication>
#include <windows.h>
#include <QtCore\QDir>
//#include "QFileDialog"
#include <QtWidgets\qfiledialog.h>

CQtFileOperate::CQtFileOperate()
{

}

CQtFileOperate::~CQtFileOperate()
{

}

QString CQtFileOperate::GetLastError()
{
    return 	m_strErrorInfo;
}
// 获取本执行程序所在的路径
QString CQtFileOperate::GetCurrentAppPath()
{
    return QCoreApplication::applicationDirPath();
}
//判断路径是否存在
bool CQtFileOperate::IsPathExist(QString strPath)
{
    if(strPath.isEmpty()||strPath.isNull())
    {
        return false;
    }
    QDir dir(strPath);
    if (dir.exists())
	{
		return true;
	}
	else
	{
		return false;
	}
}
//创建一个多层目录，如果存在就不创建
//(strPath  合法: "D:\\abc\\me" 或 "me\\you" 或 ""; 
//			不合法: "D:\\abc\\me\\" 或 "me\\you\\" 或 "\\")
bool CQtFileOperate::CreateMultiLevelPath(QString strPath)
{
    QDir dir;
    return dir.mkpath(strPath);//创建多级目录
}
//
bool CQtFileOperate::DeleteDirectory(QString strPath)
{
    if (strPath.isEmpty()||strPath.isNull())
	{
        m_strErrorInfo = "strPath Empty or Null";
		return false;
	}
    if(!IsPathExist(strPath))
    {
        return true;
    }

    QDir dir(strPath);

    dir.setFilter(QDir::AllEntries | QDir::NoDotAndDotDot);
    QFileInfoList fileList = dir.entryInfoList();
    foreach (QFileInfo fi, fileList)
    {
        if (fi.isFile())
            fi.dir().remove(fi.fileName());
        else
            DeleteDirectory(fi.absoluteFilePath());
    }
    return dir.rmpath(dir.absolutePath());
}
//拷贝文件夹
bool CQtFileOperate::CopyFolder(QString strFromPath, QString strToPath)
{
    return false;
    /*
    if(!IsPathExist(strFromPath))
    {
        return false;
    }
    if(IsPathExist(strToPath))
    {
        return false;
    }

    strFromPath.replace("/","\\");
    strToPath.replace("/","\\");

    int nLengthFrm = strFromPath.length();

	TCHAR *NewPathFrm = new TCHAR[nLengthFrm + 2];
    memcpy(NewPathFrm, strFromPath.utf16(), nLengthFrm * sizeof(TCHAR));;
	NewPathFrm[nLengthFrm] = '\0';
	NewPathFrm[nLengthFrm + 1] = '\0';



    TCHAR *szTo = new TCHAR[strToPath.length() + 2];
     memcpy(szTo, strToPath.utf16(), strToPath.length() * sizeof(TCHAR));
     szTo[strToPath.length()] = '\0';
     szTo[strToPath.length() + 1] = '\0';

	SHFILEOPSTRUCT FileOp;

	ZeroMemory((void*)&FileOp, sizeof(SHFILEOPSTRUCT));

	FileOp.fFlags = FOF_SILENT | FOF_NOCONFIRMATION;//FOF_NOCONFIRMATION ;

	FileOp.hNameMappings = NULL;

	FileOp.hwnd = NULL;

	FileOp.lpszProgressTitle = NULL;

    FileOp.pFrom = NewPathFrm;

    FileOp.pTo = szTo;

	FileOp.wFunc = FO_COPY;


    int iRet = SHFileOperation(&FileOp);
    if (iRet == 0)
	{
        delete []NewPathFrm;
        delete []szTo;
		return true;
	}
	else
	{
        delete []NewPathFrm;
        delete []szTo;
		return false;
	}
*/
	return false;
}

bool CQtFileOperate::ReNameFolder(QString strFromPath, QString strToPath)
{
    if(!IsPathExist(strFromPath))
    {
        return false;
    }

    if(IsPathExist(strToPath))
    {
        return false;
    }

    QDir dir;

    return dir.rename(strFromPath,strToPath);
}
//浏览对话框
bool CQtFileOperate::BrowseFolder(QString strInitPath, QString &strBrownPath)
{
    QString dir = QFileDialog::getExistingDirectory(0, QObject::tr("Select the directory"),strInitPath,QFileDialog::ShowDirsOnly|QFileDialog::DontResolveSymlinks);

    if(IsPathExist(dir))
    {
        strBrownPath = dir;
        return true;
    }
    else
    {
        return false;
    }
}

bool CQtFileOperate::DeleteFolderByYearMonDay(QString strFolder, int iYear, int iMonth, int iDay)
{
    QDir dir(strFolder);

    QStringList ListPath;
    if(dir.exists())
    {
        dir.setFilter(QDir::Dirs);

        QFileInfoList list = dir.entryInfoList();

        for(int j = 0;j< list.size();j++)
        {
            QFileInfo fileInfo = list.at(j);
            if (fileInfo.fileName()=="."|fileInfo.fileName()=="..")
            {
                continue;
            }
            if(fileInfo.isDir())
            {
                QString filename = fileInfo.fileName();
                QString filepath = fileInfo.filePath();

                int itmpYear = filename.mid(0,4).toInt();
                int itmpMonth = filename.mid(5,2).toInt();
                if (itmpYear < iYear)
                {
                    ListPath.push_back(filepath);

                }
                else if (itmpYear == iYear && itmpMonth <= iMonth)
                {
                    ListPath.push_back(filepath);
                }
            }
        }
    }

    for(int i = 0;i< ListPath.size();i++)
    {
         DeleteDirectory(ListPath.at(i));
    }

    return true;
}
//判断文件是否存在
bool CQtFileOperate::IsFileExist(QString strFileFullPath)
{
    if(strFileFullPath.isEmpty() | strFileFullPath.isNull())
    {
        return false;
    }
    QFileInfo fileInfo(strFileFullPath);
    if(fileInfo.isFile())
    {
        return true;
    }
    return false;
}

bool CQtFileOperate::IsFileExist(std::string strFileFullPath)
{
    QString str = QString::fromStdString(strFileFullPath);
    return IsFileExist(str);
}

bool CQtFileOperate::FileDelete(QString strFileFullPath)
{
    if(IsFileExist(strFileFullPath))
    {
        QFile File(strFileFullPath);
        if(File.exists())
        {
            return File.remove();
        }
    }
	return true;
}
//strExt = exe或者bmp  或者*
bool CQtFileOperate::FileBrowse(QString strInitPath, QString strExt, QString &strFileFullPath)
{
    QString str = "*."+strExt;
    QString FileName = QFileDialog::getOpenFileName(0,QObject::tr("Select the file"),strInitPath,str);

    if(IsFileExist(FileName))
    {
        strFileFullPath = FileName;
        return true;
    }
    else
    {
        return false;
    }
}

bool CQtFileOperate::FileSave(QString strInitPath, QString strExt, QString &strFilePath)
{
    QString str = "*."+strExt;

    QString FileName = QFileDialog::getSaveFileName(0,QObject::tr("Save as"),strInitPath,str);

    if(FileName.isEmpty() || FileName.isNull())
    {
        return false;
    }
    else
    {
        strFilePath = FileName;

        return true;
    }
}

bool CQtFileOperate::MyWritePrivateProfileString(QString strApp, QString strKey, QString strContent,QString strFilePath)
{
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrContent = (wchar_t *)strContent.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();
    return (bool)WritePrivateProfileString(lstrApp,lstrKey,lstrContent,lstrFile);
}

QString CQtFileOperate::MyReadPrivateProfileString(QString strApp, QString strKey, QString strDefault, QString strFilePath)
{
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrContent = (wchar_t *)strDefault.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();

    TCHAR Tchread[256];
    GetPrivateProfileString(lstrApp,lstrKey,lstrContent,Tchread,256,lstrFile);

    return QString::fromUtf16((ushort *)Tchread);
}

bool CQtFileOperate::MyWritePrivateProfileInt(QString strApp, QString strKey, int iContent, QString strFilePath)
{
    QString strContent = QString("%1").arg(iContent);
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrContent = (wchar_t *)strContent.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();

    return (bool)WritePrivateProfileString(lstrApp,lstrKey,lstrContent,lstrFile);
}

int CQtFileOperate::MyReadPrivateProfileInt(QString strApp, QString strKey, int iDefault, QString strFilePath)
{
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();
    return GetPrivateProfileInt(lstrApp, lstrKey, iDefault, lstrFile);
}

bool CQtFileOperate::MyWritePrivateProfileDouble(QString strApp, QString strKey,  double dbContent, QString strFilePath)
{
    QString strContent = QString("%1").arg(dbContent);
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrContent = (wchar_t *)strContent.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();

    return (bool)WritePrivateProfileString(lstrApp,lstrKey,lstrContent,lstrFile);
}

double CQtFileOperate::MyReadPrivateProfileDouble(QString strApp, QString strKey, double dbDefault, QString strFilePath)
{
    QString strContent = QString("%1").arg(dbDefault);
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrContent = (wchar_t *)strContent.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();

    TCHAR Tchread[256];
    GetPrivateProfileString(lstrApp,lstrKey,lstrContent,Tchread,256,lstrFile);
    strContent= QString::fromUtf16((ushort *)Tchread);
    return strContent.toDouble();
}

bool CQtFileOperate::MyWritePrivateProfileBool(QString strApp, QString strKey, bool bContent, QString strFilePath)
{
    QString strContent = QString("%1").arg(bContent);
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrContent = (wchar_t *)strContent.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();

    return (bool)WritePrivateProfileString(lstrApp,lstrKey,lstrContent,lstrFile);
}

bool CQtFileOperate::MyReadPrivateProfileBool(QString strApp, QString strKey, bool bDefault, QString strFilePath)
{
    QString strContent = QString("%1").arg(bDefault);
    LPCWSTR lstrApp = (wchar_t *)strApp.utf16();
    LPCWSTR lstrKey = (wchar_t *)strKey.utf16();
    LPCWSTR lstrContent = (wchar_t *)strContent.utf16();
    LPCWSTR lstrFile = (wchar_t *)strFilePath.utf16();

    TCHAR Tchread[256];
    GetPrivateProfileString(lstrApp,lstrKey,lstrContent,Tchread,256,lstrFile);
    strContent= QString::fromUtf16((ushort *)Tchread);
    return (bool)strContent.toInt();
}
//
bool CQtFileOperate::MyWritePrivateProfileString(std::string strApp, std::string strKey, std::string strContent,std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtContent= QString::fromStdString(strContent);
    QString qtFilePath = QString::fromStdString(strFilePath);

    return MyWritePrivateProfileString(qtApp,qtKey,qtContent,qtFilePath);
}

std::string CQtFileOperate::MyReadPrivateProfileString(std::string strApp, std::string strKey, std::string strDefault, std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtContent= QString::fromStdString(strDefault);
    QString qtFilePath = QString::fromStdString(strFilePath);

    QString strResult = MyReadPrivateProfileString(qtApp,qtKey,qtContent,qtFilePath);
    return strResult.toStdString();
}

bool CQtFileOperate::MyWritePrivateProfileInt(std::string strApp, std::string strKey, int iContent, std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtFilePath = QString::fromStdString(strFilePath);

    return MyWritePrivateProfileInt(qtApp,qtKey,iContent,qtFilePath);
}

int CQtFileOperate::MyReadPrivateProfileInt(std::string strApp, std::string strKey, int iDefault, std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtFilePath = QString::fromStdString(strFilePath);

    return MyReadPrivateProfileInt(qtApp,qtKey,iDefault,qtFilePath);
}

bool CQtFileOperate::MyWritePrivateProfileDouble(std::string strApp, std::string strKey,  double dbContent, std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtFilePath = QString::fromStdString(strFilePath);

    return MyWritePrivateProfileDouble(qtApp,qtKey,dbContent,qtFilePath);
}

double CQtFileOperate::MyReadPrivateProfileDouble(std::string strApp, std::string strKey, double dbDefault, std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtFilePath = QString::fromStdString(strFilePath);

    return MyReadPrivateProfileDouble(qtApp,qtKey,dbDefault,qtFilePath);
}

bool CQtFileOperate::MyWritePrivateProfileBool(std::string strApp, std::string strKey, bool bContent, std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtFilePath = QString::fromStdString(strFilePath);

    return MyWritePrivateProfileBool(qtApp,qtKey,bContent,qtFilePath);
}

bool CQtFileOperate::MyReadPrivateProfileBool(std::string strApp, std::string strKey, bool bDefault, std::string strFilePath)
{
    QString qtApp = QString::fromStdString(strApp);
    QString qtKey = QString::fromStdString(strKey);
    QString qtFilePath = QString::fromStdString(strFilePath);

    return MyReadPrivateProfileBool(qtApp,qtKey,bDefault,qtFilePath);
}

bool CQtFileOperate::WriteLog(QString strLogFilePath, QString strContent)
{
	static int iFileCount = 0;
	static int iCount = 0;//日志第几条

	if (!IsPathExist(strLogFilePath))
	{
		CreateMultiLevelPath(strLogFilePath);

		if (!IsPathExist(strLogFilePath))
		{
			return false;
		}

		iFileCount = 1;
		iCount = 0;
	}
	else
	{
		if (iFileCount <=0)//软件第一次启动时候，使用一个新文件
		{
			iFileCount = 1;
			while (true)
			{
                QString strFilePath = QString("%1/%2.log").arg(strLogFilePath).arg(iFileCount,(int)5,(int)10,QChar('0'));

                if (IsFileExist(strFilePath))
				{
					iFileCount++;
				}
				else
				{
					break;
				}
			}			
			iCount = 0;
		}
		else
		{
			if (iCount >= 2000)
			{
				iFileCount++;
				iCount = 0;
			}
		}		
	}

    QString strFilePath = QString("%1/%2.log").arg(strLogFilePath).arg(iFileCount,5,10,QChar('0'));

    QString strApp("Log");
    QString strKey;
    strKey = QString("%1").arg(iCount);
	
    MyWritePrivateProfileString(strApp, strKey, strContent, strFilePath);	
	iCount++;
	return false;
}
//获取剩余内存容量,单位M
long CQtFileOperate::GetSurplusMemorySize()
{
	MEMORYSTATUS MemoryStatus;
	GlobalMemoryStatus(&MemoryStatus);
	long lVal = (long)(MemoryStatus.dwAvailPhys / (1024 * 1024));
	return lVal;
}
// 获得当前硬盘分区的剩余容量，单位：M 输入：D:、E:、F:等
long CQtFileOperate::GetSurplusCurrentPartitionSize(QString strDrivePath)//进行修改，自动获取当前程序运行的文件路径，统计硬盘容量
{
    LPCWSTR szDrivePath = (wchar_t *)strDrivePath.utf16();

	ULARGE_INTEGER lpFreeBytesAvailableToCaller;
	ULARGE_INTEGER lpTotalNumberOfBytes;
	ULARGE_INTEGER lpTotalNumberOfFreeBytes;
    BOOL bResult = GetDiskFreeSpaceEx(szDrivePath, &lpFreeBytesAvailableToCaller, &lpTotalNumberOfBytes,
		&lpTotalNumberOfFreeBytes);
	if (bResult == FALSE)
	{
        m_strErrorInfo = QObject::tr("Failed to get the remaining capacity of the hard disk");
		return -1;
	}
	return (int)(lpTotalNumberOfFreeBytes.QuadPart / (1024 * 1024));
}
//杀死当前进程
void CQtFileOperate::ShutCurrentProgcess()
{
	UINT ExitCode = 0;
	BOOL bReturn = TRUE;
	HANDLE hCurrentProcess = GetCurrentProcess();
	bReturn = TerminateProcess(hCurrentProcess, ExitCode);
	if (bReturn == FALSE)
	{

	}
}
//////////////////////////////////////////////////////////////////////////
//用法：PrintfDebugLog("[PCB]%s starts to run\n", _func_name);
void CQtFileOperate::PrintfDebugLog(const char * strOutputString, ...)
{	
	TCHAR TstrBuffer[4096] = { 0 };
	char strBuffer[4096] = { 0 };
	va_list vlArgs;
    va_start(vlArgs, strOutputString);
	_vsnprintf_s(strBuffer, sizeof(strBuffer) - 1, strOutputString, vlArgs);
	va_end(vlArgs);

	int iLength;
	iLength = MultiByteToWideChar(CP_ACP, 0, strBuffer, strlen(strBuffer) + 1, NULL, 0);
	MultiByteToWideChar(CP_ACP, 0, strBuffer, strlen(strBuffer) + 1, TstrBuffer, iLength);

	OutputDebugString(TstrBuffer);
}

void CQtFileOperate::PrintfDebugLog(QString strOutputString)
{
    OutputDebugString((wchar_t *)strOutputString.utf16());
}

bool CQtFileOperate::Encrypt(const char szText[], unsigned int nTextLen, char szOutString[], unsigned int nOutLen)
{
	if (nTextLen <= 0 || nOutLen < nTextLen)
	{
		return false;
	}
	char chLetter;

	int i = 0;
	for (; i < nTextLen - 1; i++)
	{
		chLetter = szText[i] + i + 10;
		szOutString[i] = chLetter;
	}
	szOutString[i] = '\0';
	return true;
}

bool CQtFileOperate::Decrypt(const char szText[], unsigned int nTextLen, char szOutString[], unsigned int nOutLen)
{
	if (nTextLen <= 0 || nOutLen < nTextLen)
	{
		return false;
	}
	char chLetter;

	int i = 0;
	for (; i < nTextLen - 1; i++)
	{
		chLetter = szText[i] - i - 10;
		szOutString[i] = chLetter;
	}
	szOutString[i] = '\0';
	return true;
}

bool CQtFileOperate::SaveDIBImageToBMPFile(QString strFileName, const char * pImageBuff, long lImageWidth, long lImageHeight, long lImageBitCount)
{	
    BOOL bRet = TRUE;
    Q_ASSERT(pImageBuff != NULL && lImageWidth >= 0 && lImageHeight >= 0 && lImageBitCount > 0);

	//公共保存用的
	BITMAPINFO* m_pSaveBitmapInfo;    
    m_pSaveBitmapInfo = (BITMAPINFO *) new BYTE[sizeof(BITMAPINFO) + 255 * sizeof(RGBQUAD)];
    m_pSaveBitmapInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    m_pSaveBitmapInfo->bmiHeader.biPlanes = 1;
    m_pSaveBitmapInfo->bmiHeader.biBitCount = 0;
    m_pSaveBitmapInfo->bmiHeader.biCompression = BI_RGB;
    m_pSaveBitmapInfo->bmiHeader.biSizeImage = 0;
    m_pSaveBitmapInfo->bmiHeader.biXPelsPerMeter = 0;
    m_pSaveBitmapInfo->bmiHeader.biYPelsPerMeter = 0;
    m_pSaveBitmapInfo->bmiHeader.biClrUsed = 0;
    m_pSaveBitmapInfo->bmiHeader.biClrImportant = 0;
    m_pSaveBitmapInfo->bmiHeader.biWidth = 0;
    m_pSaveBitmapInfo->bmiHeader.biHeight = 0;
    for (int i = 0; i < 256; i++)
    {
        m_pSaveBitmapInfo->bmiColors[i].rgbBlue = (BYTE)i;
        m_pSaveBitmapInfo->bmiColors[i].rgbGreen = (BYTE)i;
        m_pSaveBitmapInfo->bmiColors[i].rgbRed = (BYTE)i;
        m_pSaveBitmapInfo->bmiColors[i].rgbReserved = 0;
    }

    m_pSaveBitmapInfo->bmiHeader.biWidth = lImageWidth;
    m_pSaveBitmapInfo->bmiHeader.biHeight = lImageHeight;
    m_pSaveBitmapInfo->bmiHeader.biBitCount = (WORD)lImageBitCount;

    wchar_t * lpFileName = (wchar_t *)strFileName.utf16();
    BOOL bRVal = TRUE;
    DWORD dwBytesRead = 0;
    DWORD dwSize = 0;
    BITMAPFILEHEADER bfh = { 0 };
    int nTable = 0;
    DWORD dwImageSize = 0;

    if (m_pSaveBitmapInfo->bmiHeader.biBitCount > 8)
    {
        nTable = 0;
    }
    else
    {
        nTable = 256;
    }

    dwImageSize = (m_pSaveBitmapInfo->bmiHeader.biWidth * m_pSaveBitmapInfo->bmiHeader.biHeight) * ((m_pSaveBitmapInfo->bmiHeader.biBitCount + 7) / 8);

    if (dwImageSize <= 0)
    {
        bRVal = FALSE;
    }
    else
    {
        bfh.bfType = (WORD)'M' << 8 | 'B';
        bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + nTable * sizeof(RGBQUAD);
        bfh.bfSize = bfh.bfOffBits + dwImageSize;

        HANDLE hFile = ::CreateFile(lpFileName,
            GENERIC_WRITE,
            0,
            NULL,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
        if (hFile == INVALID_HANDLE_VALUE) {
            bRVal = FALSE;
        }
        else {
            dwSize = sizeof(BITMAPFILEHEADER);
            ::WriteFile(hFile, &bfh, dwSize, &dwBytesRead, NULL);

            dwSize = sizeof(BITMAPINFOHEADER) + nTable * sizeof(RGBQUAD);
            ::WriteFile(hFile, m_pSaveBitmapInfo, dwSize, &dwBytesRead, NULL);

            dwSize = dwImageSize;
            WriteFile(hFile, pImageBuff, dwSize, &dwBytesRead, NULL);

            CloseHandle(hFile);
        }
    }

    if (m_pSaveBitmapInfo != NULL)
    {
        delete[]m_pSaveBitmapInfo;
        m_pSaveBitmapInfo = NULL;
    }
    if (!bRet)
    {
        m_strErrorInfo = QObject::tr("save file failed");
        return false;
    }
    return true;
}
//裁剪函数(适用于DIB图像数据)(以原图像左上角为原点, x向右, y向下)
//目标缓冲区//剪切矩形
//返回值: TRUE: 成功
//BOOL CImage::CutMe(char *pDest, CRect DestRect)
bool CQtFileOperate::CutDIBImage(char *pDest, long xDest, long yDest, long lDestWidth, long lDestHeight, const char* pSrcImageBuff, long lSrcImageWidth, long lSrcImageHeight, long lSrcImageBitCount)
{
	//注: DIB图像数据的存储方式是: 原图像的第一行存储在DIB图像数据缓冲区的最后一行, 缓冲区从下往上逐行存储.
    Q_ASSERT(pDest != NULL && pSrcImageBuff != NULL && lSrcImageWidth >= 0 && lSrcImageHeight >= 0 && lSrcImageBitCount > 0);

	int nOffset_Source = 0;//相对于缓冲区头部偏移的字节数(源)
	int nOffset_Dest = 0;  //相对于缓冲区头部偏移的字节数(目标)

	int CutPositionX = xDest;
	int CutPositionY = yDest;
	int CutWidth = lDestWidth;
	int CutHeight = lDestHeight;

	int nPixelByteCount = (lSrcImageBitCount + 7) / 8;//一个象素用几个字节表示(RGB:3)
													  //参数检查
	if ((CutPositionX + CutWidth) > lSrcImageWidth || (CutPositionY + CutHeight) > lSrcImageHeight)
	{
		m_strErrorInfo = "参数不合法";
		return false;
	}

	//进行剪切起始行的转换
	int nBeginLine = lSrcImageHeight - CutPositionY - CutHeight;

	//去到DIB图像数据缓冲区的剪切行的起始处
	nOffset_Source = (lSrcImageWidth * nBeginLine + CutPositionX) * nPixelByteCount;

	for (int i = 0; i < CutHeight; i++)//切多少行
	{
		memcpy(pDest + nOffset_Dest, pSrcImageBuff + nOffset_Source, CutWidth*nPixelByteCount);

		nOffset_Dest += (CutWidth * nPixelByteCount);  //下一行(目标)

		nOffset_Source += (lSrcImageWidth*nPixelByteCount);  //下一行(源)
	}
	return true;
}

bool CQtFileOperate::CutPlaneRGBImage(char *pDest, long xDest, long yDest, long lDestWidth, long lDestHeight, const char* pSrcImageBuff, long lSrcImageWidth, long lSrcImageHeight)
{
    Q_ASSERT(pDest != NULL && pSrcImageBuff != NULL && lSrcImageWidth >= 0 && lSrcImageHeight >= 0);

	int nOffset_Source = 0;//相对于缓冲区头部偏移的字节数(源)
	int nOffset_Dest = 0;  //相对于缓冲区头部偏移的字节数(目标)

	int CutPositionX = xDest;
	int CutPositionY = yDest;
	int CutWidth = lDestWidth;
	int CutHeight = lDestHeight;

	//参数检查
	if ((CutPositionX + CutWidth) > lSrcImageWidth || (CutPositionY + CutHeight) > lSrcImageHeight)
	{
		m_strErrorInfo = "参数不合法";
		return false;
	}

	//进行剪切起始行的转换
	int nBeginLine = lSrcImageHeight - CutPositionY - CutHeight;
	//去到DIB图像数据缓冲区的剪切行的起始处
	nOffset_Source = lSrcImageWidth * nBeginLine + CutPositionX;

	for (int i = 0; i < CutHeight; i++)//切多少行
	{
		memcpy(pDest + nOffset_Dest, pSrcImageBuff + nOffset_Source, CutWidth);					//copy一行B的数据
		memcpy(pDest + nOffset_Dest + CutHeight * CutWidth,
			pSrcImageBuff + nOffset_Source + lSrcImageWidth * lSrcImageHeight, CutWidth);		//copy一行G的数据
		memcpy(pDest + nOffset_Dest + CutHeight * CutWidth * 2,
			pSrcImageBuff + nOffset_Source + lSrcImageWidth * lSrcImageHeight * 2, CutWidth);	//copy一行R的数据

		nOffset_Dest += CutWidth;  //下一行(目标)

		nOffset_Source += lSrcImageWidth;  //下一行(源)
	}

	return true;
}

bool CQtFileOperate::ConvertRGBToPlaneR_G_B(char *pDestImageBuffR, char *pDestImageBuffG, char *pDestImageBuffB,
	const char* pSrcImageBuff, long lSrcImageWidth, long lSrcImageHeight)
{
    m_strErrorInfo = QObject::tr("This function is not available");
	return false;
}

bool CQtFileOperate::ConvertPlaneR_G_BToRGB(char* pDestImageBuff, const char *pSrcImageBuffR, const char *pSrcImageBuffG, const char *pSrcImageBuffB,
	long lSrcImageWidth, long lSrcImageHeight)
{
    m_strErrorInfo = QObject::tr("This function is not available");
	return false;
}

bool CQtFileOperate::ConvertRGBToPlaneRGB(char *pDestPlaneRGBBuff, const char* pSrcImageRGBBuff, long lSrcImageWidth, long lSrcImageHeight)
{
    m_strErrorInfo = QObject::tr("This function is not available");
	return false;
}

bool CQtFileOperate::ConvertPlaneRGBToRGB(char* pDestImageRGBBuff, const char *pSrcPlaneRGBBuff, long lSrcImageWidth, long lSrcImageHeight)
{
    m_strErrorInfo = QObject::tr("This function is not available");
	return false;
}

QString CQtFileOperate::Encrypt(const QString &strInput)
{
    if(strInput.size() == 0)
    {
        return strInput;
    }

    QString strOutput = strInput;

    for(int i = 0;i< strOutput.size();i++)
    {
         char chLetter = strOutput.at(i).toLatin1();
         chLetter = chLetter+i+10;
         QChar chOut = QChar::fromLatin1(chLetter);
         strOutput[i] = chOut;
    }
    return strOutput;
}

QString CQtFileOperate:: Decrypt(const QString &strInput)
{
    if(strInput.size() == 0)
    {
        return strInput;
    }

    QString strOutput = strInput;

    for(int i = 0;i< strOutput.size();i++)
    {

        char chLetter = strOutput.at(i).toLatin1();
        chLetter = chLetter - i - 10;
        QChar chOut = QChar::fromLatin1(chLetter);
        strOutput[i] = chOut;
    }
    return strOutput;
}
