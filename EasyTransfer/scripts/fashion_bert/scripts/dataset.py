from tqdm import tqdm 
import numpy as np 
import h5py 
from image_feature_extract import PatchFeatureExtractor, get_batch
from bert.create_pretraining_data import *
import os 

class DataPreProcess(): 
    def __init__(self) -> None:
        pass 

    @staticmethod
    def image_to_embedding(file_path : str, model_path, save_path, mode='train', num_patches=(8, 8), batch_size=32, sample_range=None):
        file_h5 = h5py.File(file_path, mode='r')
        # define the features to be retrieved
        list_of_features = ['input_image', 'input_description']
        dataset_len = len(file_h5['input_image'])  
        print("The dataset len is: {}".format(dataset_len))
        nb_batches = int(dataset_len / batch_size) 

        # get the first batch of the data
        feature_extractor = PatchFeatureExtractor(model_path)
        results = [] 
        if sample_range is not None:
            len_samples = sample_range[1] - sample_range[0]
            nb_batches = len_samples // batch_size
        
        for b in tqdm(range(0, nb_batches)): 
            b += sample_range[0] // batch_size
            features_array_batch = get_batch(file_h5, list_of_features, b, batch_size)
            imgs = features_array_batch[0] 
            results_batch = feature_extractor.predict(imgs, batch_size=batch_size, num_patches=num_patches)
            results.append(results_batch)
        # fix the remain data 
        if sample_range is None:
            features_array_batch = get_batch(file_h5, list_of_features, b, dataset_len - batch_size * (b -  1) + 1)
            imgs = features_array_batch[0] 
            results_batch = feature_extractor.predict(imgs, batch_size=batch_size, num_patches=num_patches)           
            results.append(features_array_batch)

        results = np.concatenate(results, axis=0) 
        if not os.path.exists(save_path): 
            os.makedirs(save_path) 
        np.save(save_path + 'image_embs_{}'.format(mode)+'.npy', results)
        # close the file
        file_h5.close()

    @staticmethod
    def text_to_features(file_path : str, output_file, 
                        vocab_file, do_lower_case, random_seed, max_seq_length, 
                        short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                        dupe_factor, 
                        sample_range=None):

        file_h5 = h5py.File(file_path, mode='r')
        concat_desc = file_h5['input_concat_description'] 
        
        if sample_range is not None:
            # len_samples = len( - sample_range[0]) 
            concat_desc = concat_desc[sample_range[0]: sample_range[1]]

        input_files = []
        # for input_pattern in input_file.split(","):
        #     input_files.extend(tf.gfile.Glob(input_pattern))
        input_files.append(concat_desc)

        tf.logging.set_verbosity(tf.logging.INFO)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

        rng = random.Random(random_seed)
        instances = create_training_instances(
            input_files, tokenizer, max_seq_length, dupe_factor,
            short_seq_prob, masked_lm_prob, max_predictions_per_seq,
            rng, input_array_mode=True)

        output_files = output_file.split(",")
        tf.logging.info("*** Writing to output files ***")
        for output_file in output_files:
            tf.logging.info("  %s", output_file)

        write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                        max_predictions_per_seq, output_files, save_csv=True)
                
    @staticmethod
    def merge_dataset(image_feats_path, text_feats_path, out_path, sample_range, size_patch=64, mask_img_size=5):
        img_embs = np.load(image_feats_path, allow_pickle=True)
        # img_embs = img_embs[sample_range[0]: sample_range[1]]
        num_sample = sample_range[1] - sample_range[0]
        img_embs = img_embs[:num_sample]
        text_feats = open(text_feats_path, 'r')
        
        with open(out_path, 'w') as f:
            for i in range(num_sample):
                image_mask = [1] * size_patch
                masked_patch_position = np.random.randint(0, size_patch, size=mask_img_size)
                f.writelines(','.join([str(e) for e in img_embs[i]['feature'].tolist()]) + 
                '\t' + ','.join([str(t) for t in image_mask]) + 
                '\t' +','.join([str(t) for t in masked_patch_position]) + 
                '\t' + text_feats.readline())

        text_feats.close() 

    @staticmethod
    def get_inputs_fashiongen(root_pth, mode, len_limit, batchsize=4):
        file_path = os.path.join(root_pth, "fashiongen_256_256_{}.h5".format(mode))
        model_path = "resnet_v1_50"
        save_path = os.path.join(root_pth, "fashiongen_256_256_{}_test/".format(mode))
        image_feats_path = os.path.join(root_pth, "fashiongen_256_256_{}_test/image_embs.npy".format(mode))
        vocab_file_path = "/dataset/atticus/google-bert/uncased_L-12_H-768_A-12/vocab.txt"
        text_feats_path = os.path.join(root_pth, "tmp_{}_test.csv".format(mode))
        out_path = os.path.join(root_pth, "pre{}_inputs_test".format(mode))
        batch_size = batchsize
        DataPreProcess.image_to_embedding(file_path, model_path, save_path, batch_size=batch_size, len_limit=len_limit)
        DataPreProcess.text_to_features(
            file_path, text_feats_path, 
            vocab_file=vocab_file_path, 
            do_lower_case=True, 
            random_seed=12345,
            max_seq_length=64,
            short_seq_prob=0.1,
            masked_lm_prob=0.15,
            max_predictions_per_seq=10,
            dupe_factor=5,
            len_limit=len_limit) 
        DataPreProcess.merge_dataset(image_feats_path, text_feats_path, out_path, len_limit, 64, 5) 
    
    @staticmethod
    def get_inputs_amazon(root_pth, mode, sample_range, batchsize=4):
        file_path = os.path.join(root_pth, "amazon_meta_gen_format.h5")
        model_path = "resnet_v1_50"
        save_path = os.path.join(root_pth, "pretrainval_data/")
        image_feats_path = os.path.join(root_pth, "pretrainval_data/image_embs_{}.npy".format(mode))
        vocab_file_path = "/dataset/atticus/google-bert/uncased_L-12_H-768_A-12/vocab.txt"
        text_feats_path = os.path.join(root_pth, "pretrainval_data/tmp_{}.csv".format(mode))
        out_path = os.path.join(root_pth, "pretrainval_data/pre{}_inputs".format(mode))
        DataPreProcess.image_to_embedding(file_path, model_path, save_path, mode, batch_size=batchsize, sample_range=sample_range)
        DataPreProcess.text_to_features(
            file_path, text_feats_path, 
            vocab_file=vocab_file_path, 
            do_lower_case=True, 
            random_seed=12345,
            max_seq_length=64,
            short_seq_prob=0.1,
            masked_lm_prob=0.15,
            max_predictions_per_seq=10,
            dupe_factor=5,
            sample_range=sample_range) 
        DataPreProcess.merge_dataset(image_feats_path, text_feats_path, out_path, sample_range, 64, 5) 
