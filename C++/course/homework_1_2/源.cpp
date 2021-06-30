#include <windows.h>
#include <stdio.h>
#include <iostream>

using namespace std;

int main()
{
	FILE* fp;
	RGBQUAD strPal[256];
	char* imagedata;
	char num[20] = { 0 };
	char picture[20] = { 0 };
	char picture_output[20] = { 0 };
	cout << "please input the picture number:" << endl;
	cin >> num;
	strcat_s(picture, sizeof(picture), num);
	strcat_s(picture, sizeof(picture), ".bmp");

	fopen_s(&fp, picture, "rb");//以只读方式打开bmp图片
	if (fp == 0) {
		printf("can't open the bmp file");
		exit(0);
	}
	if (fp)
	{
		BITMAPFILEHEADER filehead;//定义文件头和信息头
		BITMAPINFOHEADER infohead;
		//将图片的文件头和信息头依次放进创建的结构体中
		fread(&filehead, sizeof(BITMAPFILEHEADER), 1, fp);//size_t fread(void* restrict buffer, size_t size, size_t count, FILE * restrict stream);//C99起
		fread(&infohead, sizeof(BITMAPINFOHEADER), 1, fp);//从给定输入流stream读取最多count个对象到数组buffer中

		//将图片256个像素对应的调色板存放在strPal结构体数组中
		fread(&strPal, sizeof(RGBQUAD), 256, fp);
		//创建一个动态数组imagedata用于存放图像数据
		int height = infohead.biHeight;
		int width = infohead.biWidth;
		int MN = height * width * sizeof(BYTE);
		imagedata = new unsigned char[MN];

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				imagedata[width * i + j] = 0;
			}
		}

		//将数据读到imagedata数组中
		fread(imagedata, MN, 1, fp);//能否改成只读一次呢？
		fclose(fp);
		strcat_s(picture_output, sizeof(picture_output), num);
		strcat_s(picture_output, sizeof(picture_output), "_output.bmp");
		fopen_s(&fp, picture_output, "wb");
		if (fp == 0) {
			printf("can't open the bmp file");
			exit(0);
		}
		if (fp)
		{
			fwrite(&filehead, sizeof(BITMAPFILEHEADER), 1, fp);//size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
			fwrite(&infohead, sizeof(BITMAPINFOHEADER), 1, fp);//把ptr所指向的数组中的数据写入到给定流stream中

			//修改调色板模块，使RGB三个分量相等
			for (unsigned int num_str = 0; num_str < 256; num_str++) {
				fwrite(&strPal[num_str].rgbBlue, sizeof(BYTE), 1, fp);
				fwrite(&strPal[num_str].rgbBlue, sizeof(BYTE), 1, fp);
				fwrite(&strPal[num_str].rgbBlue, sizeof(BYTE), 1, fp);
				fwrite(&strPal[num_str].rgbReserved, sizeof(BYTE), 1, fp);
			}

			//写入图像数据
			fwrite(imagedata, MN, 1, fp);

			fclose(fp);
			printf("\n\n==================================\n");
			printf("Gray level transformation finished\n");
			printf("==================================\n\n");
			return 0;
		}
	}
}