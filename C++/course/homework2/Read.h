#pragma once
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
unsigned char* fxs;//用于存放像素的指针

unsigned char* read(const char* str)  //打开名为str的bmp文件
{
	FILE* f1;
	fopen_s(&f1,str,"rb"); 
	if (f1 == 0)
	{
		cout << "unable to open the bmp file" << endl;
	}
	if (f1)
	{
		//读位图文件头
		BITMAPFILEHEADER head;
		fread(&head, sizeof(BITMAPFILEHEADER), 1, f1);      //(void *buffer, size_t size, size_t count, FILE *stream)
		bfType = head.bfType;
		bfSize = head.bfSize;
		bfOffBits = head.bfOffBits;
		cout << "文件类型:" << head.bfType << endl;
		cout << "文件大小:" << head.bfSize << endl;
		cout << "保留字:" << head.bfReserved1 << endl;
		cout << "保留字:" << head.bfReserved2 << endl;
		cout << "偏移字节数:" << head.bfOffBits << endl;   //14+40

		//读位图信息头
		BITMAPINFOHEADER info;
		fread(&info, sizeof(BITMAPINFOHEADER), 1, f1);
		bmpWidth = info.biWidth;
		bmpHeight = info.biHeight;
		biBitCount = info.biBitCount;
		cout << "该结构的长度:" << info.biSize << endl;
		cout << "图像的宽度:" << info.biWidth << endl;
		cout << "图像的高度:" << info.biHeight << endl;
		cout << "平面数:" << info.biPlanes << endl;
		cout << "颜色位数:" << info.biBitCount << endl;
		cout << "压缩类型:" << info.biCompression << endl;
		cout << "实际位图数据占用的字节数:" << info.biSizeImage << endl;
		biXPelsPerMeter = info.biXPelsPerMeter;
		biYPelsPerMeter = info.biYPelsPerMeter;
		cout << "水平分辨率:" << info.biXPelsPerMeter << endl;
		cout << "垂直分辨率:" << info.biYPelsPerMeter << endl;
		cout << "实际使用的颜色数:" << info.biClrUsed << endl;
		cout << "重要的颜色数:" << info.biClrImportant << endl;
		lineByte = (bmpWidth * biBitCount / 8 + 3) / 4 * 4;
		MN = lineByte * bmpHeight;
		cout << "图片大小:" << MN << endl;
		fxs = new unsigned char[MN];

		//读调色板
		RGBQUAD colorTable[256];
		fread(colorTable, sizeof(RGBQUAD), 256, f1);

		//读像素
		fread(fxs, 1, MN, f1);
		fclose(f1);
	}
	return fxs;
}