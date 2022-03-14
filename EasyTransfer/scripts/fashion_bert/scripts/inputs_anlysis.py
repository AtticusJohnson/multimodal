import numpy as np
from scipy import rand 

def ana_inputs_format():
    inputs_file = "/home/atticus/proj/matching/EasyTransfer/scripts/fashion_bert/fashionbert_fashiongen_patch_train_1K"
    with open(inputs_file, "r") as f:
        for line in f.readlines(): 
            feats = line.split(",") 
            print("the format of input file is as: ", feats)
            break 



import h5py
import numpy as np
import cv2 
import os 

import json 


def ana_fashoiongen_format(): 
        
    # fashiongen_path = "/dataset/atticus/fashiongen/fashiongen_256_256_train.h5" 
    fashiongen_path = "/dataset/atticus/amazon/amazon_meta_gen_format.h5"
    BATCH_SIZE = 32
    def get_batch(file_h5, features, batch_number, batch_size=32):
        """Get a batch of the dataset
        Args:
            file_h5(str): path of the dataset
            features(list(str)): list of names of features present in the dataset
            that should be returned.
            batch_number(int): the id of the batch to be returned.
            batch_size(int): the mini-batch size
        Returns:
            A list of numpy arrays of the requested features"""
        list_of_arrays = []
        lb, ub = batch_number * batch_size, (batch_number + 1) * batch_size
        for feature in features:
            list_of_arrays.append(file_h5[feature][lb: ub])
        return list_of_arrays

    # open the file
    # file_h5 = h5py.File('fashiongen_256_256_train.h5', mode='r')
    file_h5 = h5py.File(fashiongen_path, mode='r')
    # define the features to be retrieved
    # list_of_features_fashion_gen = ['input_image', 'index', 'index_2', 'input_brand',
    # 'input_category', 'input_composition', 'input_concat_description', 'input_department',
    #  'input_description', 'input_gender', 'input_msrpUSD', 'input_pose', 'input_productID', 
    #  'input_season', 'input_subcategory']
    # list_of_features_amazon = ['asin', 'brand', 'categories', 'description', 'imUrl', 'input_concat_description', 'input_description', 'input_image', 'price', 'related', 'salesRank', 'title']

    list_of_features = ['input_image','asin', 'title', 'input_description', 'input_concat_description'] 
    dataset_len = len(file_h5['input_image']) 
    # input_description = list(file_h5['title'])
    nb_batches = int(dataset_len / BATCH_SIZE)
    
    np.random.seed(np.random.randint(0, 100))
    
    batch_nb = np.random.randint(0, nb_batches)
    # get the first batch of the data
    list_of_arrays = get_batch(file_h5, list_of_features, batch_nb, BATCH_SIZE)
    imgs = list_of_arrays[0] 
    img_id = list_of_arrays[1] 
    titles = list_of_arrays[2] 
    infos = list_of_arrays[3:] 

    if not os.path.exists("tmp"):
        os.makedirs("tmp") 
    f = open("tmp/fanshion_gen_info.txt", "w+")
    for i in range(len(imgs)):
        img = imgs[i]
        # print(img.shape)
        cv2.imwrite("tmp/{}.jpg".format(titles[i]), img) 
        for info in infos:
            m = info[i]
            f.writelines("{}\n".format(img_id[i]))
            f.writelines("{}\n".format(titles[i]))
            ss = " ".join(str(mm) for mm in m.flatten()) 
            f.writelines(ss + "\n")
            f.writelines("="*20 + "\n")
        
    # close the file
    file_h5.close()
    f.close()
    print("done") 

ana_fashoiongen_format()

import json
from typing import OrderedDict 
import pandas as pd
import gzip


def parse(path):
  g = open(path, 'r')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
    if i >= 5:
        break
  return pd.DataFrame.from_dict(df, orient='index')


def getImgUrl(path):
  img_urls = OrderedDict()
  i = 0 
  for d in parse(path):
    i += 1
    if 'imUrl' in d.keys():
      img_urls[d['asin']] = d['imUrl']
    yield img_urls
  
# img_urls = getImgUrl('meta_Clothing_Shoes_and_Jewelry.json.gz')
# print(df.iloc[4]['imUrl'])
# print("")
def ana_amazon_format():
    file_pth = "amazon_data/meta_Clothing_Shoes_and_Jewelry.json"
    #显示所有列
    pd.set_option('display.max_columns', None)
    #显示所有行
    pd.set_option('display.max_rows', None)
    #设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth',100)

    df = getDF(file_pth)
    print(df.description.tolist())

    print()

# ana_amazon_format()