#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:17:26 2018

@author: DD, UT

compute and save intermediate representations of fixations on images from the
inception_v3 net.
"""


import pickle
import keras
import os
import gc
#import exp_library as ex
import pandas as pd
import numpy as np
import re
from time import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from keras.applications.inception_v3 import preprocess_input
from modules.generators import ImageSequence

DEFAULT_SAVE_PATH_MODEL     = 'models/model_inception_v3_mixedlayeroutputs_auto.h5'
DEFAULT_IMAGE_DIRECTORY     = 'images/'
DEFAULT_SAVE_DATA_PATH      = 'activations_of_fixations_auto.p' #TODO: think about a good name maybe
DEFAULT_EYE_FIXATION_DAT    = "finalresulttab_funcsac_SFC_memory.dat"
DEFAULT_EXPERIMENT          = "memory"


all_data = pd.read_table(DEFAULT_EYE_FIXATION_DAT,encoding = "ISO-8859-1")


SAVE_PATH_MODEL = ''
IMAGE_DIRECTORY = ''
SAVE_DATA_PATH = ''

#set the fovea size
fovea= 30
#filter: get only the fixation part inside the maximum fovea possible
all_data = all_data.loc[
                      (all_data["fixposx"] >= fovea) &
                      (all_data["fixposx"] <= 1024 - fovea) &
                      (all_data["fixposy"] >= fovea) &
                      (all_data["fixposy"] <= 768 - fovea)
                      ]


def get_model(load_path = None, auto_save = True):
    """Loads or instantiates a model based on inception_v3 returning the
    outputs of all 11 mixed layers with indices from 0 to 10.

    Arguments:
        load_path: path of the already instantiated saved model.

        auto_save: if True, saves the model in the default save path, if model
        was not loaded.

    Returns:
        a keras model with all full mixed layers of inception_V3 as output
    """

    if load_path is None:
        if SAVE_PATH_MODEL == '':
            load_path = DEFAULT_SAVE_PATH_MODEL
        else:
            load_path = SAVE_PATH_MODEL
    try:
        model = keras.models.load_model(load_path)

    except OSError:
        inc_v3 = keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')

        def get_mixed_layer_names():
            layer_names = []
            for layer in inc_v3.layers:
                if 'mixed' in layer.name:
                    layer_names.append(layer.name)
            return layer_names

        mixed_layer_names = get_mixed_layer_names()

        main_mixed_layer_names = [ln for ln in mixed_layer_names if '_' not in ln]

        x = inc_v3.input
        outs = []
        for ln in main_mixed_layer_names:
            outs.append(inc_v3.get_layer(ln).output)
        model = keras.Model(inputs=x, outputs=outs)
        if auto_save:
            model.save(DEFAULT_SAVE_PATH_MODEL, include_optimizer=False)
    return model

def get_img_paths(start_dir, extensions = ['png']):
    """Returns all image paths with the given extensions in the directory.

    Arguments:
        start_dir: directory the search starts from.

        extensions: extensions of image file to be recognized.

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths


def data_selecting(data,color,masktype,maskregion,fixincalid):
    """choose the data associated to different experiment settings

    Arguments:
        data:

            ...

    Returns:
        DataFrame
    """
#masktype == 0 & maskregion == 0: Kontrollbedingung
    cleaned_data = data.loc[(data["colorimages"] == color) &
                         (data["masktype"]    == masktype) &
                         (data["maskregion"]  == maskregion) &
                         (data["fixinvalid"]  == fixincalid) ,
                         ['subject',
                          'fixposx',
                          "fixno",
                          "fixposy",
                          "imageid",
                          "masktype",
                          "maskregion",
                          "fixinvalid",
                          "colorimages"]]
    return cleaned_data

def get_eyemovements(dataFrame):
    """Take a DataFrame from Eyemovement Experiment and returns the list
       eyemovemts. Preprozessing for

    Input: Selected Experment DataFrame

    Returns: DataFrame with lists of Eyemovement

    """

    ###create list of eyemovements
    list_of_ey_x = dataFrame.groupby("imageid")["fixposx"].apply(list)
    list_of_ey_y = dataFrame.groupby("imageid")["fixposy"].apply(list)
    list_of_ey_xy = pd.concat([list_of_ey_x,list_of_ey_y], axis = 1)

    return list_of_ey_xy

model = get_model()

if IMAGE_DIRECTORY == '':
    img_paths = get_img_paths(DEFAULT_IMAGE_DIRECTORY)
else:
    img_paths = get_img_paths(IMAGE_DIRECTORY)

img_sequ = ImageSequence(paths = img_paths,
                         labels = None,
                         batch_size = 1,
                         preprocessing=preprocess_input,
                         augmentation=[])



activation_for_each_picture = []
activation_for_each_picture_SR = pd.Series()
image_paths                 = []   
experiment_info    = []               
#postloop
#loop = 0

#for i,image in enumerate(img_sequ): 
    #if i >= len(img_sequ):
    #if i >= 2:
        #break
    #loop += 1
    
#print(loop)    
    
#len(img_paths)

#THis is the for loop for only one iteration
#i=0
#iter_ = img_sequ.__iter__()
#image = iter_.__next__()
#i = 0
    
Output_df = pd.DataFrame()
    
    

for i,image in enumerate(img_sequ): 
    #if i >= len(img_sequ):
    if i >= 1:
        break
    t_start_model_predict = time()
    p = model.predict(image, batch_size = 1)
    t_stop_model_predict = time()
    
    #get the name of the picture
    picture_name = re.search("\w+(?:\.\w+)*$",img_paths[i])[0]
    
    picture_name
    #get the picture id
    #pic_id = re.search("\d+",picture_name)[0]
    
    pic_id = re.findall(r'\d+', picture_name)[0]
    pic_id
    #get the picture id
    #re.search("hp|lp",picture_name)
    
    #img_paths[1]
    
    color_type = 0
    pass_filter_type = 0
    masktype = 0
    maskregion = 0
    
    
    #Filter für die Experimentdaten
    if "color" in img_paths[i]:
        color = 1
        color_type = "color"
        #print("color")
    elif "grayscale" in img_paths[i]: 
        color = 0
        color_type = "grayscale"
        #print("greyscale")
    
    
    if DEFAULT_EXPERIMENT in img_paths[i]:
        session = 1
        #print(DEFAULT_EXPERIMENT)
    else: 
        session = 0
        #print("else")
        
        
    if "high-pass" in img_paths[i]: 
        masktype   = 2
        maskregion = 1
        pass_filter_type = "high-pass"
        #print("high-pass")
    elif "low-pass" in img_paths[i]: 
        masktype   = 1
        maskregion = 1
        pass_filter_type = "low-pass"
        #print("low-pass")
    elif "original" in img_paths[i]: 
        masktype = 0
        maskregion = 0
        pass_filter_type = "original"
        #print("original")
        
    
    #print("-----------")
    exp_control_color = data_selecting(all_data,
                                         color,
                                         masktype,
                                         maskregion,
                                         fixincalid = 0)
    
    #preprocessing
    list_of_ey_xy = get_eyemovements(exp_control_color)
    
    #experiment_type = re.sub("[_|[0-9]]*","",picture_name_only)
    
    #print("picid",pic_id)
    
    #get the eyemovements from picture id
    eyemovements =  list_of_ey_xy.loc[list_of_ey_xy.index == int(pic_id), :]
    
    
    #Orignal Picture Size
    img_h, img_w = image.shape[1], image.shape[2]
    
    #ĺayer count
    layer_count = len(p)
    #looping inside the layer
    list_of_activations = []
    
    list_of_activations_SR = pd.Series()
    
    
    if(not eyemovements.empty and session ==1):
        t_start_get_all_layers = time()
        for layer in range(layer_count):
            #debug
            if layer >= 3:
                break
            
            #inside a channel
            #layershape
            part_h   = p[layer][0].shape[0]
            part_w   = p[layer][0].shape[1]
            #number of channels
            channels = p[layer][0].shape[2]
            #scale factors for the particular feature
            scale_h = img_h / part_h
            scale_w = img_w / part_w
                       
    
            #scaled fovea
            scaled_fovea_y = round(fovea / scale_h)
            scaled_fovea_x = round(fovea / scale_w)
            activations = []

            
            t_start_get_activation_eyemovements = time()
            #get the activations from each channel with eye movements
            fix_activation = []
            
            fix_sum = []
            for channel in range(channels):
                
                    if channel >= 4:
                        break
                    channel_act = []
                    for fix in range(len(eyemovements.iloc[0,1])):
                        activations = []
                        
                        #save the mean of the sliced partial activaions on the scaled 
                        #fixation position
                        #save the activations and append it for each channel
                        #scaled fixations for the particular feature
                        scaled_fix_y = int(eyemovements.iloc[0,1][fix] / scale_h)
                        scaled_fix_x = int(eyemovements.iloc[0,0][fix] / scale_w)
                        activations.append(p[layer][0][ scaled_fix_y - scaled_fovea_y : 
                                                        scaled_fix_y + scaled_fovea_y + 1 ,
                                                        scaled_fix_x - scaled_fovea_x : 
                                                        scaled_fix_x + scaled_fovea_x + 1 ,
                                                        channel]
                                                        .mean())
                        fix_sum.append(fix)  

                    
                    fix_activation.append(activations)   
                    
            
            list_of_activations_SR = list_of_activations_SR.append(pd.Series(fix_activation), ignore_index=True)
            
            #print(row)
            
            #save the activations for each layer
            #list_of_activations.append(fix_ac)
            
            
            t_stop_get_activation_eyemovements = time()
        
        
        
        #Output_df = Output_df.append([row[0]],ignore_index=True)
        activation_for_each_picture_SR = activation_for_each_picture_SR.append(list_of_activations_SR, ignore_index=True)
        #save the activations for each 
        activation_for_each_picture.append(list_of_activations)
        image_paths.append(img_paths[i])
        experiment_info.append([color_type,pass_filter_type, pic_id])
        t_stop_get_all_layers = time()
        print("one step")
    else:
        pass
        print("Leer")
    
    
    
#p[5][0][21 - 2 : 21 + 2 + 1 ,
#        54 - 2 : 54 + 2 + 1 , 1].mean()

#eyemovements.iloc[0,1][2]  
#eyemovements.iloc[0,0][2]     
#p[5][0].shape  
# =============================================================================
#     for fix in range(len(eyemovements.iloc[0,1])):
#         activations = []
#         for channel in range(channels):
#             channel_act = []
#             
#             #save the mean of the sliced partial activaions on the scaled 
#             #fixation position
#             #save the activations and append it for each channel
#             #scaled fixations for the particular feature
#             scaled_fix_y = int(eyemovements.iloc[0,1][fix] / scale_h)
#             scaled_fix_x = int(eyemovements.iloc[0,0][fix] / scale_w)
#             activations.append(p[layer][0][ scaled_fix_y - scaled_fovea_y : 
#                                             scaled_fix_y + scaled_fovea_y + 1 ,
#                                             scaled_fix_x - scaled_fovea_x : 
#                                             scaled_fix_x + scaled_fovea_x + 1 ,
#                                             channel]
#                                             .mean())
#         fix_sum.append(fix)  
#   
# 
#         
# 
# 
#             ##Output_df.append([row],ignore_index=True)
#             
#             #channels_act.append(activations)
#         
#         fix_activation.append(activations)   
# row=pd.Series(fix_activation,fix_sum)
# 
# #save the activations for each layer
# list_of_activations.append(fix_activation)
# =============================================================================
    
    
#p[0][0].shape    

#this could be a dataframe
total = {'paths'             : image_paths,
         'activations'       : activation_for_each_picture,
         'experiment_info'   : experiment_info
         }


#pickle.dump( total, open( "saved/save_all_2.p", "wb" ) )


t_start_get_loading = time()

gc.disable()
#aved_activation_list = pickle.load( open( "save_all.p", "rb" ) )
t_stop_get_loading = time()
gc.enable()

#len(saved_activation_list[1,0])
#oaded = pd.read_pickle("save_all.p")

#loaded.head()


print ('the function model_predict takes %f' %(t_stop_model_predict-t_start_model_predict))
print ('the function getlayer takes %f' %(t_stop_get_activation_eyemovements-t_start_get_activation_eyemovements))
print ('the function getactivation from eyefixation takes %f' %(t_stop_get_all_layers - t_start_get_all_layers))
print ('the function loading filtakes %f' %(t_stop_get_loading - t_start_get_loading))

#python has a problem with pickle.dump and very large files
def write(data, file_path):
    """Writes data (using pickle) to a file in smaller byte chunks.

    Arguments:
        data: the data to be pickled

        file_path: the file to be written to (or created)
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(total)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

if SAVE_DATA_PATH == '':
    write(total, DEFAULT_SAVE_DATA_PATH)
else:
    write(total, SAVE_DATA_PATH)