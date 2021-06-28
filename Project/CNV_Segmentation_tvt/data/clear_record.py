import os
import glob

delete_selection = 'unet' # 网络名小写
checkpath = "../checkpoints"
for f in os.listdir(checkpath):
    if delete_selection=="all":
        for checkpoint in os.listdir(checkpath+os.sep+f):
            os.remove(os.path.join(checkpath,f,checkpoint))
    else:
        if f==delete_selection:
            for checkpoint in os.listdir(checkpath+os.sep+f):
                os.remove(os.path.join(checkpath,f,checkpoint))

delete_file = list()
logpath = "../logs"
if os.path.exists(os.path.join(logpath,delete_selection+".txt")):
    delete_file.append(os.path.join(logpath,delete_selection+".txt"))
if os.path.exists(os.path.join(logpath,delete_selection+"_best_indicator.txt")):
    delete_file.append(os.path.join(logpath,delete_selection+"_best_indicator.txt"))
for i in range(len(delete_file)):
    print("deleting",delete_file[i])
    os.remove(delete_file[i])