#include<iostream>
#include <windows.h>
#include <fstream>
#include "Read.h"
#include "Write.h"

using namespace std;

unsigned char* fxs1;
unsigned char* fxs2;
unsigned char* fxs3;

int main()
{
	//读取原始图信息
	fxs1=read("1.bmp"); //读出图像的数据文件

	//打开新建文件，存入模糊图像
	FILE* f2;
	fopen_s(&f2,"blur.bmp", "wb");
	write(f2); //写入除图像数据外的其他信息

	//运动模糊
	int x0, y0, T = 10, x = 10, y = 10;
	float a = 0.1, b = 0, sum, temp;
	fxs2 = new unsigned char[MN];
	for (int i = 0; i < bmpHeight; i++)
	{
		for (int j = 0; j < lineByte; j++)
		{
			x0 = 0;
			y0 = 0;
			sum = 0;
			temp = 0;

			for (int t = 0; t < T; t++)
			{
				x0 = i * lineByte + j - t * x / T;  //水平
				y0 = i * lineByte + j - t * y * lineByte / T;  //垂直
				x0 <= 0 ? 0 : x0;
				y0 <= 0 ? 0 : y0;
				temp = fxs1[x0] * a + fxs1[y0] * b;
				sum += temp; // 积分累加
			}
			sum = sum > 255 ? 255 : sum;
			fxs2[i * lineByte + j] = sum;
		}
	}
	fwrite(fxs2, MN, 1, f2);
	fclose(f2);

	//打开新建文件,存入恢复图像
	FILE* f3;
	fopen_s(&f3,"deblur.bmp", "wb");
	write(f3);
	int i, j, temp1, temp2, n, m, B = 10, totalq, q, q1, q2, z, p, A = 150;
	//int x = 10;// 赋近似值常量与移动距离
	int K = lineByte / x; //取 K 值
	cout << "K：" << K << endl;
	int error[10];
	fxs3 = new unsigned char[MN];
	for (j = 0; j < bmpHeight; j++)
	{
		for (i = 0; i < 10; i++)
		{
			error[i] = 0;
			for (n = 0; n < K; n++)
				for (m = 0; m <= n; m++)
				{
					if (i == 0 && m == 0)//判断像素是否为一行的开始处
					{
						temp1 = temp2 = 0;
					}
					else // 进行差分运算
					{
						temp1 = fxs2[lineByte * j + m * 10 + i];
						temp2 = fxs2[lineByte * j + m * 10 + i - 1];
					}
					error[i] = error[i] + temp1 - temp2;
				}
			error[i] = B * error[i] / K;
		}
		for (i = 0; i < lineByte; i++)
		{
			m = i / x;// 计算 m 与 z
			//cout << m << endl;
			z = i - m * x;
			totalq = 0;// 初始化
			q = 0;
			for (n = 0; n <= m; n++)
			{
				q1 = i - x * n;
				if (q1 == 0)
					q = 0;
				else// 进行差分运算
				{
					q2 = q1 - 1;
					temp1 = fxs2[lineByte * j + q1];
					temp2 = fxs2[lineByte * j + q2];
					q = (temp1 - temp2) * B;
				}
				totalq = totalq + q;
			}
			p = error[z];
			temp1 = totalq + A - p;// 得到 f(x,y)的值
			if (temp1 < 0)
				temp1 = 0;
			if (temp1 > 255)
				temp1 = 255; // 判断像素的取值符合范围
			fxs3[lineByte * j + i] = temp1;
		}
	}
	fwrite(fxs3, MN, 1, f3);
}
