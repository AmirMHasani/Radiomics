#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import pydicom
import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import random 
import radiomics
from radiomics import featureextractor
import cv2 
import shutil 
import copy
import matplotlib.pyplot as plt  


# In[2]:


path = r"Y:\AMprj\Projects\Ongoing Projects\Original Research\Amir Hasani Projects\LAM Projects\LAM Prognosis"


# In[3]:


os.chdir(path)


# In[4]:


input_data_csv = "input_data.csv"
only_image_path = "onlyimages" # Save only the images without the masks here 
threshold_background = 100.0001 # threshold to get just the lungs background, needed to check the cysts near the bounary
threshold_cyst = 20.0001 # It is used to get just the cysts using the 3rd chanel on the masked image 
crop_v=[150,350] # Cropping the relevent section only
crop_h=[100,430] 
kernel = np.ones((9, 9), np.uint8) # kernel to dilate the cysts masks 
min_area = 5 # Get only cysts larger than this 
area_background_min = 200 # Rmove any regions of lungs smaller than this from background masks 
max_area = 500000 # 
red_line_thickness = 6 # to draw the red lines on the cysts 
img_size = 512 # Image size 
show_info = False # Show some info to debug the code 
output_file_without_dilation = "output_without_dilation.csv" # output file without dilation to save the data 
output_file_with_dilation = "output_with_dilation.csv" # output file with dilation to save the data 
overwrite = True # Generate new output file, 


if(overwrite):
    if(os.path.exists(output_file_without_dilation)):
        os.remove(output_file_without_dilation)
    if(os.path.exists(output_file_with_dilation)):
        os.remove(output_file_with_dilation)


# In[ ]:


def get_binary_image(image,threshold):
    """
    image: RGB image of the mask sumperimposed onto mask 
    threshold: Threshold used to binarize the 3rd chanel of the image. 
    
    return: binary image 
    """
    binary = copy.deepcopy(image[:,:,2])
    binary[binary==0] = 255
    binary[binary<threshold] = 0
    binary[binary>threshold] = 255
    binary = 255 - binary
    return binary 

def get_instances(image):
    """
    image_path: full path to the image, 
    threshold: 0-255 value which can be used to segement the images using thresholding. 
    
    return: boundary_points: Can be used to draw red line 
            instances: get all instances (cysts) present in the image  
    """
    contours, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    instances = []
    boundary_points = []
    img = copy.deepcopy(image)
    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        if (area >min_area and area < max_area): # Ignoring small regions. 
            mask = np.zeros(img.shape, dtype="uint8")
            mask = cv2.fillPoly(mask,[cnt],255)
            instances.append(mask)
            boundary_points.append(cnt)
    return boundary_points, instances

def remove_small_regions(image):
    """
    image: bacgrkound image,
    remove small regions in the background image which  has area < are_background_min 
    """
    contours, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    instances = []
    img = copy.deepcopy(image)
    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        if (area < area_background_min): # Ignoring small regions. 
            image = cv2.fillPoly(image,[cnt],255)
    return image

def boundary_check(background,instance):
    instance = background*instance
    return instance

def process_masked_image(img_path,save_path="red_region.png",with_dilation=False):
    """
    Input:  masked image (either of left lung or right lung)

    Return: Final red region areas of all the cysts present in th image   
    """
    ds = pydicom.dcmread(img_path)
    image = ds.pixel_array.astype(np.uint8)
    img_shape = image.shape

    
    binary_cyst = get_binary_image(image=image,
                              threshold=threshold_cyst)
    binary_cyst_dilated  = cv2.dilate(binary_cyst , kernel, iterations=1)
    
    
    all_cysts_this_image = binary_cyst_dilated-  binary_cyst
    
    
    ds = pydicom.dcmread(img_path)
    image = ds.pixel_array.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image*(1-binary_cyst/255)
    cv2.imwrite(f"{save_path}_image.png",image)
    cv2.imwrite(f"{save_path}_label.png",all_cysts_this_image)

    return image, all_cysts_this_image


def process_patient(left_lung_path,right_lung_path,process_lung,patient_id,with_dilation):
    """
    left_lung_path: Path to the left lung masked image 
    right_lung_path: Path to the right lung masked image 
    process_lung: Which lung to process to obtain the red region 
    patient_id: patient id for which the data is processed. Both the lung masked image belong to the same paient
    with_dilation: Process with out without dilation of cysts 
    Return: Features of the image and cyst masks 
    """
    AXIALL = sorted(glob.glob(f"{left_lung_path}*"))
    AXIALR = sorted(glob.glob(f"{right_lung_path}*"))
    if(process_lung =="right" or process_lung=="Right"):
        images_to_process = AXIALR
    else:
        images_to_process = AXIALL

    if (show_info):
        print(100*"-")
        print("Step1: Getting full image without mask ".center(100,"-"))
        print(f"Found {len(AXIALL)} masked images of left lung in {left_lung_path} path ")
        print(f"Found {len(AXIALR)} masked images of right lung in {right_lung_path} path")
        print(f"Processing masked images from both lungs to get the full image without mask ")
        print(100*"-")

    all_cysts_all_images = np.zeros(([len(AXIALR),img_size,img_size])) # [N,512,512]
    all_images = np.zeros(([len(AXIALR),img_size,img_size])) # [N,512,512]

    for idx, img_path in enumerate(images_to_process):
        img_name = AXIALR[idx].split("\\")[-1]
        image, all_cysts_this_image = process_masked_image(img_path,
                                                    save_path=f"{only_image_path}\\{img_name}_{process_lung}",
                                                    with_dilation=with_dilation) # Get all cysts from right Lung of image
        all_cysts_all_images[idx,:,:] = all_cysts_this_image
        all_images[idx,:,:] = image


    if (show_info):
        print("Step3: Extracting features ".center(100,"-"))
        print(f"Now we have 3D data of the patient in the form of images {all_images.shape} and labels {all_cysts_all_images.shape}")
        print(f"We can now extract features of the data using all the {all_images.shape[0]} slices from a patient")
        print(100*"-")

    image = sitk.GetImageFromArray(all_images)
    label = sitk.GetImageFromArray(all_cysts_all_images/255) # labels 0 and 1 only accepted 
    features_dict = {}
    # Instantiate a PyRadiomics feature extractor object
    extractor = featureextractor.RadiomicsFeatureExtractor()
    result = extractor.execute(image,label)

    features_dict[f"{patient_id}"] = {**dict({"Image_title":f"{process_lung}_Lung"}),**result}

    return result,features_dict


# In[10]:


# Process data of just one patient, for both the lungs and with and without dilation 
if(os.path.exists(only_image_path)):
    shutil.rmtree(only_image_path)
os.mkdir(only_image_path)


# In[15]:

input_data = pd.read_excel(input_data_csv,names=["Patient_ID","LeftLungPath","RightLungPath"])
all_patient_data = []

for i in range(len(input_data)):
    patient_id = input_data.iloc[i]["Patient_id"]
    AXIALL_path = input_data.iloc[i]["LeftLungPath"] # Right lung path which contains masked image 
    AXIALR_path = input_data.iloc[i]["RightLungPath"] # Right lung path which contains masked image 
    os.mkdir(os.path.join(only_image_path,str(patient_id)))
    for idx,(with_dilation,output_file) in enumerate(zip([True],
                                                            [output_file_with_dilation])):
            os.mkdir(os.path.join(only_image_path,str(patient_id),"Left_Lung"))
            result , features_dict = process_patient(left_lung_path=AXIALL_path,
                                                    right_lung_path=AXIALR_path,
                                                    process_lung="Left",
                                                    patient_id=patient_id,
                                                    with_dilation=with_dilation)
            all_patient_data.append(features_dict)
            os.mkdir(os.path.join(only_image_path,str(patient_id),"Right_Lung"))
            result , features_dict = process_patient(left_lung_path=AXIALL_path,
                                                    right_lung_path=AXIALR_path,
                                                    process_lung="Right",
                                                    patient_id=patient_id,
                                                    with_dilation=with_dilation)
            all_patient_data.append(features_dict)
            

with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        headers = list(result.keys())
        writer.writerow(['Patient_ID',"Image_title"] + headers)
        for patient_data in all_patient_data:
            for mask_name in patient_data.keys():
                features = patient_data[mask_name]
                row = [mask_name,features["Image_title"]]
                for feature_name in headers:
                        row.append(features[feature_name])
                writer.writerow(row)



