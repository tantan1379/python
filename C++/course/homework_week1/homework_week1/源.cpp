#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <windows.h>
#include <fstream>

using namespace std;


typedef struct
{
	BITMAPFILEHEADER file;
	BITMAPINFOHEADER info;
}bmp;

void showFileHeader(BITMAPFILEHEADER fh);
void showInfoHeader(BITMAPINFOHEADER ih);

int main()
{
	char num[20] = { 0 };
	char picture[20] = { 0 };
	FILE* fp;
	cout << "please input the picture number:" << endl;
	cin >> num;
	strcat_s(picture, sizeof(picture), num);
	strcat_s(picture, sizeof(picture), ".bmp");
	fopen_s(&fp, picture, "rb");
	if (fp == 0)
	{
		printf("can't open the bmp image.\n ");
		exit(0);
	}
	BITMAPFILEHEADER fileheader;
	BITMAPINFOHEADER infoheader;
	fread(&fileheader, sizeof(BITMAPFILEHEADER), 1, fp);
	fread(&infoheader, sizeof(BITMAPINFOHEADER), 1, fp);
	showFileHeader(fileheader);
	showInfoHeader(infoheader);
	//bmp m;
	//m = readbmpfile();
	//printf("%d", sizeof(unsigned short int));

	return 0;
}

void showFileHeader(BITMAPFILEHEADER fh)
{
	ofstream outfile("fileheader.txt", ios::out | ios::trunc);
	cout << "--------" << endl;
	cout << "位图文件头：" << endl;
	outfile << "位图文件头：" << endl;
	cout << "文件类型：" << fh.bfType << endl;
	outfile << "文件类型：" << fh.bfType << endl;
	cout << "文件大小:" << fh.bfSize << endl;
	outfile << "文件大小:" << fh.bfSize << endl;
	cout << "保留字_1:" << fh.bfReserved1 << endl;
	outfile << "保留字_1:" << fh.bfReserved1 << endl;
	cout << "保留字_2:" << fh.bfReserved2 << endl;
	outfile << "保留字_2:" << fh.bfReserved2 << endl;
	cout << "实际位图数据的偏移字节数:" << fh.bfOffBits << endl;
	outfile << "实际位图数据的偏移字节数:" << fh.bfOffBits << endl;
}

void showInfoHeader(BITMAPINFOHEADER ih)
{
	ofstream outfile("infoheader.txt", ios::out | ios::trunc);
	cout << "--------" << endl;
	cout << "位图信息头:" << endl;
	outfile << "位图信息头:" << endl;
	cout << "图片宽度（像素）:" << ih.biWidth << endl;
	outfile << "图片宽度（像素）:" << ih.biWidth << endl;
	cout << "图片高度（像素）:" << ih.biHeight << endl;
	outfile << "图片高度（像素）:" << ih.biHeight << endl;
	cout << "颜色位数:" << ih.biBitCount << endl;
	outfile << "颜色位数:" << ih.biBitCount << endl;
	cout << "实际位图数据占用的字节数:" << ih.biSizeImage << endl;
	outfile << "实际位图数据占用的字节数:" << ih.biSizeImage << endl;
	cout << "实际使用的颜色数:" << ih.biClrUsed << endl; //等于0时表示有2^biBitCount个颜色索引表
	outfile << "实际使用的颜色数:" << ih.biClrUsed << endl;
	cout << "重要的颜色数:" << ih.biClrImportant << endl;
	outfile << "重要的颜色数:" << ih.biClrImportant << endl;
}
//bmp readbmpfile()
//{
//	bmp m;
//	FILE* fp;
//	if ((fp=fopen("1.bmp","rb")) ==NULL)
//	{
//		printf("can't open the bmp image.\n ");
//		exit(0);
//	}
//
//	else
//	{
//		fread(&m.file.bfType, sizeof(m.file.bfType),1, fp);
//		printf("类型为%s\n", m.file.bfType);
//		/*fread(&m.file.bfType, sizeof(char), 1, fp);
//		printf("%c\n",m.file.bfType);*/
//		fread(&m.file.bfSize, sizeof(long), 1, fp);
//		printf("文件长度为%d\n", m.file.bfSize);
//		fread(&m.file.bfReserverd1, sizeof(short int), 1, fp);
//		printf("保留字1为%d\n", m.file.bfReserverd1);
//		fread(&m.file.bfReserverd2, sizeof(short int), 1, fp);
//		printf("保留字2为%d\n", m.file.bfReserverd2);
//		fread(&m.file.bfOffBits, sizeof(long), 1, fp);
//		printf("偏移量为%d\n", m.file.bfOffBits);
//		fread(&m.info.biSize, sizeof(long), 1, fp);
//		printf("此结构大小为%d\n", m.info.biSize);
//		fread(&m.info.biWidth, sizeof(long), 1, fp);
//		printf("位图的宽度为%d\n", m.info.biWidth);
//		fread(&m.info.biHeight, sizeof(long), 1, fp);
//		printf("位图的高度为%d\n", m.info.biHeight);
//		fread(&m.info.biPlanes, sizeof(short), 1, fp);
//		printf("目标设备位图数%d\n", m.info.biPlanes);
//		fread(&m.info.biBitcount, sizeof(short), 1, fp);
//		printf("颜色深度为%d\n", m.info.biBitcount);
//		fread(&m.info.biCompression, sizeof(long), 1, fp);
//		printf("位图压缩类型%d\n", m.info.biCompression);
//		fread(&m.info.biSizeImage, sizeof(long), 1, fp);
//		printf("位图大小%d\n", m.info.biSizeImage);
//		fread(&m.info.biXPelsPermeter, sizeof(long), 1, fp);
//		printf("位图水平分辨率为%d\n", m.info.biXPelsPermeter);
//		fread(&m.info.biYPelsPermeter, sizeof(long), 1, fp);
//		printf("位图垂直分辨率为%d\n", m.info.biYPelsPermeter);
//		fread(&m.info.biClrUsed, sizeof(long), 1, fp);
//		printf("位图实际使用颜色数%d\n", m.info.biClrUsed);
//		fread(&m.info.biClrImportant, sizeof(long), 1, fp);
//		printf("位图显示中比较重要颜色数%d\n", m.info.biClrImportant);
//	}
//	return m;
//}