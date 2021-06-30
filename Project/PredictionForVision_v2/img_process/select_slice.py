import os
import shutil
import SimpleITK as sitk
import glob


if __name__ == "__main__":
    ori_path = "F:\\Dataset\\AMD\\AMD_Origin_3d_in_need\\"
    des_path = "F:\\Dataset\\AMD\\AMD_TimeSeries_CL\\3_single_origin\\"


    pat_index = "076"
    v1_index = 6
    v2_index = 6
    v3_index = 6
    if not os.path.exists(des_path+pat_index):
        os.mkdir(des_path+pat_index)
    if not os.path.exists(des_path+pat_index+"/"+pat_index+"_v1"):
        os.mkdir(des_path+pat_index+"/"+pat_index+"_v1")
    if not os.path.exists(des_path+pat_index+"/"+pat_index+"_v2"):
        os.mkdir(des_path+pat_index+"/"+pat_index+"_v2")
    if not os.path.exists(des_path+pat_index+"/"+pat_index+"_v3"):
        os.mkdir(des_path+pat_index+"/"+pat_index+"_v3")
    des_pat_path = des_path+pat_index

    v1 = os.path.join(ori_path,pat_index,pat_index+"_v1",pat_index+"_v1.nii.gz")
    v2 = os.path.join(ori_path,pat_index,pat_index+"_v2",pat_index+"_v2.nii.gz")
    v3 = os.path.join(ori_path,pat_index,pat_index+"_v3",pat_index+"_v3.nii.gz")
    
    v1_array = sitk.GetArrayFromImage(sitk.ReadImage(v1))
    v2_array = sitk.GetArrayFromImage(sitk.ReadImage(v2))
    v3_array = sitk.GetArrayFromImage(sitk.ReadImage(v3))
    
    v1_res = sitk.GetImageFromArray(v1_array[v1_index:v1_index+4])
    v2_res = sitk.GetImageFromArray(v2_array[v1_index:v1_index+4])
    v3_res = sitk.GetImageFromArray(v3_array[v1_index:v1_index+4])

    sitk.WriteImage(v1_res,des_pat_path+"/"+pat_index+"_v1"+"/"+pat_index+"_v1.nii.gz")
    sitk.WriteImage(v2_res,des_pat_path+"/"+pat_index+"_v2"+"/"+pat_index+"_v2.nii.gz")
    sitk.WriteImage(v3_res,des_pat_path+"/"+pat_index+"_v3"+"/"+pat_index+"_v3.nii.gz")