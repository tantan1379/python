#pragma once
#pragma warning(disable:4996)
#include <iostream>
#include <windows.h>
#include <fstream>
using namespace std;
int bfType, bfSize, bfOffBits;;
int bmpWidth;//图像的宽
int bmpHeight;//图像的高
int biBitCount;//图像类型，每像素位数（3*8）
int MN;
int lineByte;  //每一行的字节数（4的倍数）
int biXPelsPerMeter, biYPelsPerMeter;//分辨率
unsigned char *fxs1, *fxs2, *fxs3,*fxs4;   //用于存放像素的指针
void read(const char* str)
{
	FILE *f1 = fopen(str, "rb");  //只读方式打开，文件必须存在
	if (f1 == 0)
	{
		cout << "打开失败！" << endl;
	}
	if (f1)
	{
		cout << "打开成功！" << endl;

		//读位图文件头
		BITMAPFILEHEADER file;
		fread(&file, sizeof(BITMAPFILEHEADER), 1, f1);      //(void *buffer, size_t size, size_t count, FILE *stream)
		bfType = file.bfType;
		bfSize = file.bfSize;
		bfOffBits = file.bfOffBits;
		cout << "文件类型:" << file.bfType << endl;
		cout << "文件大小:" << file.bfSize << endl;
		cout << "保留字" << file.bfReserved1 << endl;
		cout << "保留字" << file.bfReserved2 << endl;
		cout << "偏移字节数" << file.bfOffBits << endl;   //14+40

		//读位图信息头
		BITMAPINFOHEADER info;     
		fread(&info, sizeof(BITMAPINFOHEADER), 1, f1);
		bmpWidth = info.biWidth;
		bmpHeight = info.biHeight;
		biBitCount = info.biBitCount;
		cout << "该结构的长度" << info.biSize << endl;
		cout << "图像的宽度" << info.biWidth << endl;
		cout << "图像的高度" << info.biHeight << endl;
		cout << "平面数" << info.biPlanes << endl;
		cout << "颜色位数" << info.biBitCount << endl;
		cout << "压缩类型" << info.biCompression << endl;
		cout << "实际位图数据占用的字节数" << info.biSizeImage << endl;
		biXPelsPerMeter = info.biXPelsPerMeter;
		biYPelsPerMeter = info.biYPelsPerMeter;
		cout << "水平分辨率" << info.biXPelsPerMeter << endl;
		cout << "垂直分辨率" << info.biYPelsPerMeter << endl;
		cout << "实际使用的颜色数" << info.biClrUsed << endl;
		cout << "重要的颜色数" << info.biClrImportant << endl;
		lineByte = (bmpWidth *biBitCount / 8 + 3) / 4 * 4;
		MN = lineByte * bmpHeight;
		cout << "图片大小: " << MN << endl;
		fxs1 = new unsigned char[MN];

		//读调色板
		RGBQUAD colorTable[256];
		fread(colorTable, sizeof(RGBQUAD), 256, f1);

		//读像素
		fread(fxs1, 1, MN, f1);   
		fclose(f1);
	}
}