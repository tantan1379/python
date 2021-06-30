import SimpleITK as sitk
import shutil
from PIL import Image

if __name__ =='__main__':
    ori_img ="F:/dicom/mc177/176210/360443/1429625/00000753"
    des_path ="F:/Dataset/AMD/AMD_TimeSeries_CL/0_to_18_single/002/V1-OCT/"
    img_array = sitk.GetArrayFromImage(sitk.ReadImage(ori_img))
    # print(img_array.shape)
    for i,one_img in enumerate(img_array):
        img = Image.fromarray(one_img)
        img.save(des_path+"杨益良"+('0'+str(i) if i<10 else str(i))+".jpg")