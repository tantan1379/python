import SimpleITK as sitk
import os
import glob
import numpy as np
import shutil

root = "F:\\Dataset\\gliomas\\"
if not os.path.exists(root+'data'):
    os.mkdir(root+"data")
if not os.path.exists(root+'label'):
    os.mkdir(root+"label")
# raw_path = "D:\\BaiduCloud\\3DT1"
# des_path = "F:\\Dataset\\gliomas\\data\\"
label_raw_path = "D:\\BaiduCloud\\roi_204\\"
label_des_path = "F:\Dataset\\gliomas\\label\\"


if __name__ == '__main__':
    # # 原始数据复制并改名
    # for img in os.listdir(raw_path):
    #     print(img[4:7])
    #     shutil.copyfile(os.path.join(raw_path,img),os.path.join(des_path,"volume-"+img[4:7]+".nii"))

    # 标签数据整合，并去除rroi部分
    pat_all = []
    pat_sort_all = []
    for pat in os.listdir(label_raw_path):
        pat_all += glob.glob(os.path.join(label_raw_path, pat))

    for pat in pat_all:
        one_pat = []
        one_pat += glob.glob(os.path.join(label_raw_path, pat, "roi*"))
        pat_sort_all.append(one_pat)

    for i, pat in enumerate(pat_sort_all):
        num = pat[0].split(os.sep)[-2][4:7]
        if(int(num) > 274):
            print(num)
            for iter, label in enumerate(pat):
                img = sitk.ReadImage(label, sitk.sitkInt16)
                img_array = sitk.GetArrayFromImage(img)
                if iter == 0:
                    label_sum = np.zeros(img_array.shape)
                label_sum += img_array
            # label_sum[label_sum > 1] = 1
            out_img = sitk.GetImageFromArray(label_sum)
            sitk.WriteImage(out_img, os.path.join(
                label_des_path, "segmentation-"+num+".nii"))
