from six import b, text_type
from image_feature_extract import batch_images
from scripts.dataset import DataPreProcess 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# os.system("")

def get_inputs():
    # file_path = "/dataset/atticus/fashiongen_256_256_validation.h5"
    model_path = "resnet_v1_50"
    # save_path = "./data/fashiongen_256_256_validation/"
    file_path = "amazon_data/tmp_test.h5"
    save_path = "amazon_data/"
    len_limit = 20000
    batch_size = 4 
    DataPreProcess.image_to_embedding(file_path, model_path, save_path, batch_size=batch_size, len_limit=len_limit)
    DataPreProcess.text_to_features(
        '/dataset/atticus/fashiongen_256_256_validation.h5', './tmp.csv', 
        vocab_file="/dataset/atticus/google-bert/uncased_L-12_H-768_A-12/vocab.txt", 
        do_lower_case=True, 
        random_seed=12345,
        max_seq_length=64,
        short_seq_prob=0.1,
        masked_lm_prob=0.15,
        max_predictions_per_seq=10,
        dupe_factor=5,
        len_limit=len_limit)
    DataPreProcess.merge_inputs(len_limit, 64, 5) 

## check if all line are same 
# file_path = "/home/atticus/proj/matching/EasyTransfer/scripts/fashion_bert/fashionbert_fashiongen_patch_train_1K"
# with open(file_path, 'r') as f:
#     cnt_same_org = 0 
#     line_ = ""
#     for line in f.readlines(): 
#         if line.strip().split('\t')[0] == line_:
#             cnt_same_org += 1 
#         line_ = line.strip().split('\t')[0] 

# import numpy as np 
# img_embs = np.load("data/fashiongen_256_256_validation/image_embs.npy", allow_pickle=True)
# cnt_same = 0
# for i in range(0, img_embs.shape[0] - 1):
#     if (img_embs[i]['feature'] == img_embs[i+1]['feature']).all():
#         cnt_same += 1 
# print("adjcent same img embs is: ", cnt_same) 

DataPreProcess.get_inputs_amazon(root_pth="/dataset/atticus/amazon", mode="val", sample_range=(75000, 76000) , batchsize=8)

## check if mine and official inputs data are as the same 
# my_inputs = open("pretrain_inputs", 'r')  
# official_inputs = open("fashionbert_fashiongen_patch_train_1K", 'r') 
# import numpy as np 
# list_lines = [] 
# cnt_samples = 0 
# for line in my_inputs.readlines():
#     list_lines.append(line) 
#     cnt_samples += 1 


# print("the total count of inputs is: ", cnt_samples)  
# my_line = my_inputs.readline() 

# print(0)
# off_l = official_inputs.readline()
# my_list = my_line.strip().split('\t')
# my_arr = [l.split(',') for l in my_list]
# print(len(my_line.split(',')))

# spec_conds = [] 
# img_info = off_l.split('\t')[:3]

# for i in range(len(off_l)):
#     if not off_l[i].replace('.','',1).isdigit():
#         spec_conds.append((i, off_l[i]))

# my_list = np.asarray([line.split(',') for line in my_inputs.readlines()])
# official_list = np.asarray([line.split(',') for line in official_inputs.readlines()])
# print("end")
