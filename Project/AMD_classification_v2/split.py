import os
import sys
import glob
import pandas as pd
import random
import shutil 

root = "F:\\Lab\\AMD_CL\\preprocessed\\"
target = "F:\\Lab\\AMD_CL\\split\\"
param = ["train","val","test"]
images,labels,image_label = [],[],[]
count=0
for i in range(3):
    if not os.path.exists(os.path.join(target,param[i])):
        os.mkdir(os.path.join(target,param[i]))

for i in range(1,4):
    for j in range(3):
        if not os.path.exists(os.path.join(target,param[j],str(i))):
            os.mkdir(os.path.join(target,param[j],str(i)))

# put all images' path into a list
for c in os.listdir(root):
    images+=glob.glob(os.path.join(root,c,"*.jpg"))
    images+=glob.glob(os.path.join(root,c,"*.png"))
    images+=glob.glob(os.path.join(root,c,"*.jpeg"))
# put all the label corresponding to the image into a list
for img in images:
    labels.append(img.split(os.sep)[-2])

# using pd to create a frame
frame = pd.DataFrame({"image":images,"label":labels})

for _,row in frame.iterrows():
    image_label.append((row["image"],row["label"]))

random.shuffle(image_label)
for img,label in image_label:
    if(label=="1"):
        count+=1
total_image = len(image_label)
train_list = image_label[:int(total_image*0.6)]
val_list = image_label[int(total_image*0.6):int(total_image*0.8)]
test_list = image_label[int(total_image*0.8):]

for img,label in train_list:
    shutil.copyfile(img, os.path.join(target,"train",label,img.split(os.sep)[-1]))
for img,label in val_list:
    shutil.copyfile(img, os.path.join(target,"val",label,img.split(os.sep)[-1]))
for img,label in test_list:
    shutil.copyfile(img, os.path.join(target,"test",label,img.split(os.sep)[-1]))