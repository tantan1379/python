#include<iostream>
#include <windows.h>
#include <fstream>
#include "Read.h"
#include "Write.h"

using namespace std;


int main()
{
	unsigned char* fxs1;
	unsigned char* fxs2;
	float nk[256] = { 0 };
	int sum[256] = { 0 };
	char num[20] = { 0 };
	char picture[20] = { 0 };
	char picture_output[20] = { 0 };
	int i, j, k, m, sk=0;

	//打开原始文件
	cout << "please input the picture number:" << endl;
	cin >> num;
	strcat_s(picture, sizeof(picture), num);
	strcat_s(picture, sizeof(picture), ".bmp");
	//读取名为str的字符串（文件名），并返回该图片的像素数据
	fxs1=read(picture);

	//新建文件	
	FILE* f2;
	strcat_s(picture_output, sizeof(picture_output), num);
	strcat_s(picture_output, sizeof(picture_output), "_output.bmp");
	fopen_s(&f2, picture_output, "wb");
	if (f2 == 0)
	{
		cout << "unable to create the bmp file!" << endl;
	}
	if (f2)
	{
		//将除图像数据以外全部写入文件(调色板r=g=b显示灰度图)，并让文件指针停留在图像数据开头
		write(f2);
		//原始图片像素值输出到文件，求nk
		ofstream outfile("origin_pix.txt", ios::out | ios::trunc);
		for (int i = 0; i < MN; i++)
		{
			m = fxs1[i];
			outfile << m << ",";//将每个像素点的像素值以逗号隔开，保存在txt文件内
			if ((i + 1) % 10 == 0)//保存格式为每行显示10个像素点的值
				outfile << endl;

			for (int j = 0; j < 256; j++)//计算未均衡化的直方图序列（统计图片中所有的像素点在各个灰度级的个数）
			{
				if (m==j)
				{
					nk[j]++;
					break;
				}
			}
		}
		cout << "\n\n原始图片像素已输出到文件\n" << endl;

		//均衡化
		fxs2 = new unsigned char[MN];
		for (int k = 0; k < 256; k++)
		{
			for (j = 0; j <= k; j++)
				sum[k] = sum[k] + nk[j];//计算累积分布函数
		}
		for (i = 0; i < MN; i++)
		{
			for (j = 0; j < 256; j++)
				if (fxs1[i]==j)
				{
					sk = (int)(255 *sum[j]/MN);//公式，乘以255是因为像素值是以0-255保存的，0-1无法显示
					break;
				}
			sk >= 255 ?255: sk;
			sk <= 0 ? 0 : sk;
			fxs2[i] = sk;//将均衡化后的像素值保存在新文件的图像数据中
		}

		//均衡化后像素输出到文件
		ofstream outfile2("output_pix.txt", ios::out | ios::trunc);
		for (int i = 0; i < MN; i++)          //（255,255,255）
		{
			m = fxs1[i];
			outfile2 << m << ",";
			if ((i + 1) % 10 == 0)
				outfile2 << endl;
		}
		cout << "\n均衡化后图片像素已输出到文件\n" << endl;

		fwrite(fxs2, MN, 1, f2);
		fclose(f2);
	}
}