import os

# root = r"F:\Dataset\CNV_Seg\png_split_tvt"
# for f in os.listdir(root):
#     f_path = root+os.sep+f
#     for t in os.listdir(f_path):
#         t_path = f_path+os.sep+t
#         for one_pat in os.listdir(t_path):
#             one_pat_path = t_path+os.sep+one_pat
#             for iter,one_pic in enumerate(os.listdir(one_pat_path)):
#                 os.rename(os.path.join(one_pat_path,one_pic),os.path.join(one_pat_path,one_pat+"_"+str(iter+1)+".png"))

# root = r"F:\Dataset\CNV_Seg\png"
# for f in os.listdir(root):
#     f_path = root+os.sep+f
#     for one_pat in os.listdir(f_path):
#         one_pat_path = f_path+os.sep+one_pat
#         for iter,one_pic in enumerate(os.listdir(one_pat_path)):
#             os.rename(os.path.join(one_pat_path,one_pic),os.path.join(one_pat_path,one_pat+"_"+str(iter+1)+".png"))

root =r"F:\Dataset\CNV_Seg\png_split_tvt\all_mask"
for tvt in os.listdir(root):
    tvt_path = root+os.sep+tvt
    for one_pat in os.listdir(tvt_path):
        one_pat_path = tvt_path+os.sep+one_pat
        for iter,one_png in enumerate(sorted(os.listdir(one_pat_path))):
            print(iter,one_png)