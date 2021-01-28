#define WIDTHBYTES(bits) (((bits) + 31) / 32 * 4);
#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#include<iostream>
#include <windows.h>
#include <fstream>
#include <time.h>  
#include "Read.h"
#include "Write.h"
using namespace std;

BOOL fourn(double* data/*psrc*/, unsigned long nn[]/*w*/, int ndim/*2*/, int isign)
{
    int idim;
    unsigned long i1, i2, i3, i2rev, i3rev, ip1, ip2, ip3, ifp1, ifp2;
    unsigned long ibit, k1, k2, n, nprev, nrem, ntot;
    double tempi, tempr;
    double theta, wi, wpi, wpr, wr, wtemp;

    for (ntot = 1, idim = 1; idim <= ndim; idim++)
        ntot *= nn[idim];
    nprev = 1;
    for (idim = ndim; idim >= 1; idim--) {
        n = nn[idim];
        nrem = ntot / (n * nprev);
        ip1 = nprev << 1;
        ip2 = ip1 * n;
        ip3 = ip2 * nrem;
        i2rev = 1;
        for (i2 = 1; i2 <= ip2; i2 += ip1) {
            if (i2 < i2rev) {
                for (i1 = i2; i1 <= i2 + ip1 - 2; i1 += 2) {
                    for (i3 = i1; i3 <= ip3; i3 += ip2) {
                        i3rev = i2rev + i3 - i2;
                        SWAP(data[i3], data[i3rev]);
                        SWAP(data[i3 + 1], data[i3rev + 1]);
                    }
                }
            }
            ibit = ip2 >> 1;
            while (ibit >= ip1 && i2rev > ibit) {
                i2rev -= ibit;
                ibit >>= 1;
            }
            i2rev += ibit;
        }
        ifp1 = ip1;
        while (ifp1 < ip2) {
            ifp2 = ifp1 << 1;
            theta = isign * 6.28318530717959 / (ifp2 / ip1);
            wtemp = sin(0.5 * theta);
            wpr = -2.0 * wtemp * wtemp;
            wpi = sin(theta);
            wr = 1.0;
            wi = 0.0;
            for (i3 = 1; i3 <= ifp1; i3 += ip1) {
                for (i1 = i3; i1 <= i3 + ip1 - 2; i1 += 2) {
                    for (i2 = i1; i2 <= ip3; i2 += ifp2) {
                        k1 = i2;
                        k2 = k1 + ifp1;
                        tempr = wr * data[k2] - wi * data[k2 + 1];
                        tempi = wr * data[k2 + 1] + wi * data[k2];
                        data[k2] = data[k1] - tempr;
                        data[k2 + 1] = data[k1 + 1] - tempi;
                        data[k1] += tempr;
                        data[k1 + 1] += tempi;
                    }
                }
                wr = (wtemp = wr) * wpr - wi * wpi + wr;
                wi = wi * wpr + wtemp * wpi + wi;
            }
            ifp1 = ifp2;
        }
        nprev *= n;
    }
    return true;
}

BOOL WINAPI WienerDIB(unsigned char *lpDIBBits, LONG lWidth, LONG lHeight)
{
    // 指向源图像的指针
    unsigned char* lpSrc;
    //循环变量
    long i;
    long j;
    //像素值
    unsigned char pixel;
    // 图像每行的字节数
    LONG lLineBytes;
    //用于做FFT的数组
    double* fftSrc, * fftKernel, * fftNoise;
    double a, b, c, d, e, f, multi;
    //二维FFT的长度和宽度
    unsigned long nn[3];
    //图像归一化因子
    double MaxNum;

    // 计算图像每行的字节数
    lLineBytes = WIDTHBYTES(lWidth * 8);

    double dPower = log((double)lLineBytes) / log(2.0);
    if (dPower != (int)dPower)
    {
        return false;
    }
    dPower = log((double)lHeight) / log(2.0);
    if (dPower != (int)dPower)
    {
        return false;
    }

    fftSrc = new double[lHeight * lLineBytes * 2 + 1];
    fftKernel = new double[lHeight * lLineBytes * 2 + 1];
    fftNoise = new double[lHeight * lLineBytes * 2 + 1];

    nn[1] = lHeight;
    nn[2] = lLineBytes;
    for (j = 0; j < lHeight; j++)
    {
        for (i = 0; i < lLineBytes; i++)
        {
            // 指向源图像倒数第j行，第i个象素的指针  
            lpSrc = (unsigned char*)lpDIBBits + lLineBytes * j + i;

            pixel = (unsigned char)*lpSrc;

            fftSrc[(2 * lLineBytes) * j + 2 * i + 1] = (double)pixel;
            fftSrc[(2 * lLineBytes) * j + 2 * i + 2] = 0.0;

            if (i < 5 && j == 0)
            {
                fftKernel[(2 * lLineBytes) * j + 2 * i + 1] = 1 / 5.0;
            }
            else
            {
                fftKernel[(2 * lLineBytes) * j + 2 * i + 1] = 0.0;
            }
            fftKernel[(2 * lLineBytes) * j + 2 * i + 2] = 0.0;
            if (i + j == ((int)((i + j) / 8)) * 8)
            {
                fftNoise[(2 * lLineBytes) * j + 2 * i + 1] = -16.0;
            }
            else
            {
                fftNoise[(2 * lLineBytes) * j + 2 * i + 1] = 0.0;
            }
            fftNoise[(2 * lLineBytes) * j + 2 * i + 2] = 0.0;
        }
    }

    srand((unsigned)time(NULL));
    //对源图像进行FFT
    fourn(fftSrc, nn, 2, 1);
    //对卷积核图像进行FFT
    fourn(fftKernel, nn, 2, 1);
    //对噪声图像进行FFT
    fourn(fftNoise, nn, 2, 1);

    for (i = 1; i < lHeight * lLineBytes * 2; i += 2)
    {
        a = fftSrc[i];
        b = fftSrc[i + 1];
        c = fftKernel[i];
        d = fftKernel[i + 1];
        e = fftNoise[i];
        f = fftNoise[i + 1];
        multi = (a * a + b * b) / (a * a + b * b - e * e - f * f);
        if (c * c + d * d > 1e-3)
        {
            fftSrc[i] = (a * c + b * d) / (c * c + d * d) / multi;
            fftSrc[i + 1] = (b * c - a * d) / (c * c + d * d) / multi;
        }
    }

    //对结果图像进行反FFT
    fourn(fftSrc, nn, 2, -1);

    //确定归一化因子
    MaxNum = 0.0;
    for (j = 0; j < lHeight; j++)
    {
        for (i = 0; i < lLineBytes; i++)
        {
            fftSrc[(2 * lLineBytes) * j + 2 * i + 1] =
                sqrt(fftSrc[(2 * lLineBytes) * j + 2 * i + 1] * fftSrc[(2 * lLineBytes) * j + 2 * i + 1]\
                    + fftSrc[(2 * lLineBytes) * j + 2 * i + 2] * fftSrc[(2 * lLineBytes) * j + 2 * i + 2]);
            if (MaxNum < fftSrc[(2 * lLineBytes) * j + 2 * i + 1])
                MaxNum = fftSrc[(2 * lLineBytes) * j + 2 * i + 1];
        }
    }

    //转换为图像
    for (j = 0; j < lHeight; j++)
    {
        for (i = 0; i < lLineBytes; i++)
        {
            // 指向源图像倒数第j行，第i个象素的指针  
            lpSrc = (unsigned char*)lpDIBBits + lLineBytes * j + i;

            *lpSrc = (unsigned char)(fftSrc[(2 * lLineBytes) * j + 2 * i + 1] * 255.0 / MaxNum);
        }
    }

    delete[] fftSrc;
    delete[] fftKernel;
    delete[] fftNoise;
    // 返回
    return true;
}

int main()
{
	//读取原始图信息
	read("1.bmp");

	//打开新建文件，存入模糊图像
	FILE* f2 = fopen("blurred.bmp", "wb");
	write(f2);

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
	FILE* f3 = fopen("restored.bmp", "wb");
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

    FILE* f4 = fopen("3.bmp", "wb");
    write(f4);
    fxs4 = new unsigned char[MN];
    WienerDIB(fxs4, bmpWidth, bmpHeight);
    fwrite(fxs4, MN, 1, f4);
}
