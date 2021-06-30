#pragma once
#include <iostream>
#include <windows.h>
#include <fstream>
#include "Read.h"
using namespace std;
void write(FILE* f)
{
	BITMAPFILEHEADER fileHead;
	fileHead.bfType = 0x4D42;  //ASCII值表示BM
	fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + MN;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;
	fileHead.bfOffBits = 54;
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, f);

	BITMAPINFOHEADER fileinfo;
	fileinfo.biBitCount = biBitCount;
	fileinfo.biClrImportant = 0;
	fileinfo.biClrUsed = 0;
	fileinfo.biCompression = 0;
	fileinfo.biHeight = bmpHeight;
	fileinfo.biPlanes = 1;
	fileinfo.biSize = 40;
	fileinfo.biSizeImage = MN;
	fileinfo.biWidth = bmpWidth;
	fileinfo.biXPelsPerMeter = biXPelsPerMeter;    //分辨率不变
	fileinfo.biYPelsPerMeter = biYPelsPerMeter;
	fwrite(&fileinfo, sizeof(BITMAPINFOHEADER), 1, f);

	RGBQUAD colorTable[256];
	for (int i = 0; i < 256; i++)
	{
		colorTable[i].rgbBlue = (BYTE)i;
		colorTable[i].rgbGreen = (BYTE)i;
		colorTable[i].rgbRed = (BYTE)i;
	}
	fwrite(colorTable, sizeof(RGBQUAD), 256, f);
}