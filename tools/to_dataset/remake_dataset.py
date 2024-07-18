#!/usr/bin/env python
# coding: utf-8

# In[1]:


import platform, sys, os
import glob


# # Install WorkingWithFiles
# To install WorkingWithFiles go to next link https://github.com/trucomanx/WorkingWithFiles
# 

# # Variables

# In[2]:



## Model of network
#model_type = 'mobilenet_v3';
model_type = 'efficientnet_b3'
#model_type = 'inception_v3';
#model_type = 'inception_resnet_v2';
#model_type = 'resnet_v2_50';


categories=['angry','disgusted','fearful','happy','neutral','sad','surprised'];



# # If command line

# In[3]:


#print('cmd entry:', sys.argv)

for n in range(len(sys.argv)):
    if sys.argv[n]=='--model':
        model_type=sys.argv[n+1];
    if sys.argv[n]=='--times':
        times=int(sys.argv[n+1]);

print('model_type:',model_type)

## Input   
input_dir = '/mnt/boveda/DATASETs/FACE-EMOTION/mcfer_v1.0/archive/test/neutral';
print('input_dir:',input_dir)

## Output
output_dir = '/mnt/boveda/DATASETs/FACE-EMOTION/mcfer_extras/tool_custom_output';
print('output_dir:',output_dir)


################################################################################
# # Biblioteca Local
sys.path.append('../library');
import Classifier as mylib


# # Bibliotecas externas
import PIL
import WorkingWithFiles as rnfunc
from tensorflow.keras.preprocessing.image import load_img

#
from tqdm import tqdm as TQDM


# # Classifier
Clf=mylib.FaceEmotionClassifier(model_type);


# # Create directory
try: 
    os.makedirs(output_dir) 
except: 
    pass



# # Testing people



pattern=os.path.join(input_dir,'*.png');
total_list=glob.glob(pattern);


L0=len(total_list);


for n in TQDM(range(L0)):
    img=load_img(total_list[n]);
    basename=os.path.basename(total_list[n]);
    res=Clf.from_img_pil(img);
    
    local_output=os.path.join(output_dir,categories[res]);
    try: 
        os.makedirs(local_output);
    except: 
        pass;
    filename_out=os.path.join(local_output,basename);
    os.rename(total_list[n], filename_out)



