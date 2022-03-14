
# image resize to 256^2 
# info include description 
from traceback import print_tb
import pandas as pd 
import os 
import h5py 
import cv2 
from tqdm import tqdm 
import numpy as np
import json 

str_dtype = h5py.special_dtype(vlen=str)

def get_file_list(pth):
    return os.listdir(pth) 

def image_proccess(): 
    pass 

def write_h5_file(file_name, dataset):
 
    #创建文件并写入数据
    f = h5py.File(file_name, 'w')
    for k, v in dataset.items():
        if isinstance(v[0], np.ndarray):
            f.create_dataset(k, data=np.stack(v, axis=0)) 
        elif isinstance(v[0], dict):
            v = [json.dumps(n) for n in v]
            f.create_dataset(k, (len(asciiList), 1), 'S400', data=asciiList)
        elif isinstance(v[0], str):
            asciiList = [n.encode("ascii", "ignore") for n in v]
            f.create_dataset(k, (len(asciiList), 1), 'S400', data=asciiList)
        elif isinstance(v[0], float):
            f.create_dataset(k, data=np.asarray(v))
        elif isinstance(v[0], int):
            f.create_dataset(k, data=np.asarray(v))
        elif isinstance(v[0], list):
            asciiList = [";".join(n[0]).encode("ascii", "ignore") for n in v]
            f.create_dataset(k, (len(asciiList), 1), 'S400', data=asciiList)
        else:
            raise "type error !"
    f.close() 


# def info_proccess():


#     img_pth = ""
#     amazon_src_pth = ""
#     img_names = set([n.strip().split('.')[0] for n in get_file_list(img_pth)]) 
    
#     # df = getDF(amazon_src_pth)

#     # feature_names = df.columns()
#     # print("the feature_names are as follows: \n", feature_names)

def parse(path):
  g = open(path, 'rb')
  for l in g:
    yield eval(l)


def info_proccess(amazon_src_pth, img_saved_path):
    # info must include description 
    # asin as number, img name extracted from imUrl 
    N = 1503384

    img_names_from_dir = get_file_list(img_saved_path)
    saved_data = {}
    saved_data['input_image'] = []
    saved_data['input_description'] = []
    saved_data['input_concat_description'] = []
    cnt_pass = 0 
    FEATS = set(['asin', 'title', 'related', 'price', 'salesRank', 'brand', 'categories', 'description', 'imUrl'])
    for idx, amazon_src_file in tqdm(enumerate(parse(amazon_src_pth)), total=N):
        # if idx >= 1000: 
        #     break 
        
        # amazon_src_file = eval(amazon_src_file)
        # item_img_id = amazon_src_file['asin']
        # item_title = amazon_src_file['title']
        # item_price = amazon_src_file['price'] 
        # item_sales_rank = amazon_src_file['salesRank'] 
        # item_brand = amazon_src_file['brand'] 
        # item_categories = amazon_src_file['categories'] 
        # item_img_url = amazon_src_file['imUrl']
        # item_img_name = item_img_url.strip().split("/")[-1].split(".")[0] + '.jpg'
        item_img_name = '{}.jpg'.format(amazon_src_file['asin'])
        feature_names = amazon_src_file.keys()
        if (item_img_name not in img_names_from_dir) or ('description' or 'asin' or 'title') not in feature_names:
            # print(item_img_name)
            cnt_pass += 1 
            continue

        item_img_name = '{}.jpg'.format(amazon_src_file['asin'])
        item_description = amazon_src_file['description']
        # print(amazon_src_file.keys())

        if len(item_description) < 5:
            cnt_pass += 1 
            continue      
        try:
            img = cv2.resize(cv2.imread(os.path.join(img_saved_path, item_img_name)), (256, 256))
        except:
            cnt_pass += 1 
            continue
        
        for feature_name in FEATS:
            if feature_name not in saved_data.keys():
                if feature_name not in feature_names:
                    if feature_name == 'price':
                        saved_data[feature_name] = [-1]
                    else:
                        saved_data[feature_name] = ['']
                else:
                    saved_data[feature_name] = [amazon_src_file[feature_name]]
            else: 
                if feature_name not in feature_names:
                    if feature_name == 'price':
                        saved_data[feature_name].append(-1)
                    else:
                        saved_data[feature_name].append('')
                else:
                    saved_data[feature_name].append(amazon_src_file[feature_name])     
        
        saved_data['input_description'].append(item_description)
        saved_data['input_concat_description'].append(item_description)
        saved_data['input_image'].append(img) 
    
    print("pass number is: ", cnt_pass)

    write_h5_file("/dataset/atticus/amazon/amazon_meta_gen_format.h5", saved_data)

        
amazon_src_pth = "amazon_data/meta_Clothing_Shoes_and_Jewelry.json"
img_saved_path = "/dataset/atticus/amazon/imgs"

info_proccess(amazon_src_pth, img_saved_path)


def merge_to_h5py():
    pass 

