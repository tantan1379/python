#include <Windows.h>
#include <iostream>

typedef struct tagIMAGEDATA
{
	BYTE blue;///8位灰度图用其中1个
	//BYTE green;
	//BYTE red;
}IMAGEDATA;

using namespace std;
int main()
{
	char num[20] = { 0 };
	char picture[20] = { 0 };
	char picture_output_1[20] = { 0 };
	char picture_output_2[20] = { 0 };
	cout << "please input the picture number:" << endl;
	cin >> num;

	strcat_s(picture, sizeof(picture), num);
	strcat_s(picture, sizeof(picture), ".bmp");
	FILE* fp;
	RGBQUAD strPla[256];//256色调色板

	fopen_s(&fp, picture, "rb");
	if (fp == 0) {
		printf("文件打开失败！\n");
	}
	BITMAPFILEHEADER fileHead;//定义位图文件头结构
	fread(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);
	fseek(fp, sizeof(BITMAPFILEHEADER), 0);//跳过位图文件头结构
	BITMAPINFOHEADER infoHead;//定义位图信息头结构
	fread(&infoHead, sizeof(BITMAPINFOHEADER), 1, fp);
	cout << "图像高度为:" << infoHead.biHeight << endl;
	cout << "图像宽度为:" << infoHead.biWidth << endl;



	fread(strPla, sizeof(RGBQUAD), 256, fp);

	IMAGEDATA* imagedata = NULL; //动态分配存储原图片的像素信息的二维数组
	IMAGEDATA* imagedataReverse = NULL;//动态分配存储裁剪后的图片的像素信息的二维数组
	IMAGEDATA* imagedataScal = NULL;

	int height, width;
	height = infoHead.biHeight;
	width = infoHead.biWidth;
	imagedata = (IMAGEDATA*)malloc(height * width * sizeof(IMAGEDATA));//为原始图像分配内存空间
	imagedataReverse = (IMAGEDATA*)malloc(height * width * sizeof(IMAGEDATA));//为原始图像分配内存空间

	//初始化原始图片的像素数组
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			(*(imagedata + i * width + j)).blue = 0;
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			(*(imagedataReverse + i * width + j)).blue = 0;
		}
	}

	//读出图片的像素数据，读取时一次一行，读高度数的行
	fread(imagedata, sizeof(struct tagIMAGEDATA) * width, height, fp);
	fclose(fp);

	//图片缩放
	float ExpScalValue; int ZoomToWidth; int ZoomToHeight;
	cout << "\n请输入要缩放的倍数" << endl;
	cin >> ExpScalValue;

	ZoomToHeight = (int)(ExpScalValue * height);
	ZoomToWidth = (int)(ExpScalValue * width);
	ZoomToWidth = (ZoomToWidth * sizeof(IMAGEDATA) + 3) / 4 * 4;

	cout << "缩放后图像高度为: " << ZoomToHeight << endl;
	cout << "缩放后图像宽度为: " << ZoomToWidth << endl;


	imagedataScal = (IMAGEDATA*)malloc(sizeof(IMAGEDATA) * ZoomToHeight * ZoomToWidth);
	//初始化缩放后的像素数组
	for (int i = 0; i < ZoomToHeight; i++) {
		for (int j = 0; j < ZoomToWidth; j++) {
			(*(imagedataScal + ZoomToWidth * i + j)).blue = 0;
		}
	}

	int pre_i, pre_j, after_i, after_j;
	for (int i = 0; i < ZoomToHeight; i++) {
		for (int j = 0; j < ZoomToWidth; j++) {
			after_i = i;
			after_j = j;
			pre_i = (int)(after_i / ExpScalValue);
			pre_j = (int)(after_j / ExpScalValue);
			if (pre_i <= height && pre_i >= 0 && pre_j <= width && pre_j >= 0) {
				*(imagedataScal + ZoomToWidth * i + j) = *(imagedata + width * pre_i + pre_j);
			}
		}
	}
	//保存bmp图片
	strcat_s(picture_output_1, sizeof(picture_output_1), num);
	strcat_s(picture_output_1, sizeof(picture_output_1), "_zoom.bmp");
	if (fopen_s(&fp, picture_output_1, "wb") != 0) {
		cout << "文件创建失败代码：" << GetLastError();
		cout << "create the bmp file error!" << endl;
		return NULL;
	}
	BITMAPFILEHEADER zoomfileheader;
	BITMAPINFOHEADER zoominfoheader;
	zoomfileheader = fileHead;
	zoomfileheader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 1024 + ZoomToHeight * ZoomToWidth;
	fwrite(&zoomfileheader, sizeof(BITMAPFILEHEADER), 1, fp);
	zoominfoheader = infoHead;
	zoominfoheader.biHeight = ZoomToHeight;
	zoominfoheader.biWidth = ZoomToWidth;
	fwrite(&zoominfoheader, sizeof(BITMAPINFOHEADER), 1, fp);

	fwrite(strPla, sizeof(RGBQUAD), 256, fp);

	for (int i = 0; i < ZoomToHeight; i++) {
		for (int j = 0; j < ZoomToWidth; j++) {
			fwrite(&((*(imagedataScal + i * ZoomToWidth + j)).blue), sizeof(BYTE), 1, fp);
		}
	}
	printf("\n\n缩放完成\n");
	fclose(fp);

	strcat_s(picture_output_2, sizeof(picture_output_2), num);
	strcat_s(picture_output_2, sizeof(picture_output_2), "_reverse.bmp");
	if (fopen_s(&fp, picture_output_2, "wb") != 0) {
		cout << "文件创建失败代码：" << GetLastError();
		cout << "create the bmp file error!" << endl;
		return NULL;
	}

	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&infoHead, sizeof(BITMAPINFOHEADER), 1, fp);
	fwrite(strPla, sizeof(RGBQUAD), 256, fp);


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			(*(imagedataReverse + width * i + j)).blue = 255 - (*(imagedata + width * i + j)).blue;
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			fwrite(&((*(imagedataReverse + i * width + j)).blue), sizeof(BYTE), 1, fp);
		}
	}
	printf("\n\n反相完成\n");
	fclose(fp);
}
