#pylint: disable-all
import tensorflow as tf
import numpy as np
from google.cloud import storage
import subprocess
import random
import os



def parse_image(example_proto,  image_features_dict):
    return tf.io.parse_single_example(example_proto, image_features_dict)


def select_tiles_on_classRatio(ds_np_gen , class_ratio=0.5, img_size = 256*256):
    """ takes a numpy iterator of img_dic of tiles and selects only img tiles that have higher
    class ratio than the threshold"""
    thr = class_ratio * img_size
    for img_dic in  ds_np_gen:
        img = img_dic['cwf']
        if np.count_nonzero(img) > thr:
            yield img_dic

def np_to_tfr_save(ds_gen, file_name="./test_tfRecord.gz"):
    with tf.io.TFRecordWriter(file_name,options=tf.io.TFRecordOptions(
    compression_type='GZIP')) as writer:
        for img_dic in ds_gen:
            
            feature = {}
            for k, v in img_dic.items():
                if k == 'cwf':
                    feature[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v.flatten()))
                else:
                    feature[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v.flatten()))                
            
            # Construct the Example proto object
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize the example to a string
            serialized = example.SerializeToString()

            # write the serialized objec to the disk
            writer.write(serialized)

def write_file_to_gs(file_name,bucket_name,blob_name):
    json_file = get_json_config()
    client = storage.Client.from_service_account_json(json_file)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_name)


def get_json_config():
    json_file = os.path.join(os.path.abspath('..') , "gc_service_account.json")
    return json_file


def delete_file_from_disk(file_name):
    cmd = f"rm {file_name}"
    status, _ = subprocess.getstatusoutput(cmd)
    return status


def list_gen(processed_tfr_files_list, num_files, image_features_dict, shuffle = True):
    gen_list = []
    for file in processed_tfr_files_list[:num_files]:
        image_dataset = tf.data.TFRecordDataset(file, 
                                                compression_type='GZIP')

        ds = image_dataset.map(lambda proto: parse_image(proto, image_features_dict), 
                               num_parallel_calls=5) 
        ds_np_gen = ds.as_numpy_iterator()
        gen_list.append(ds_np_gen)
        if shuffle:
            random.shuffle(gen_list)
    return gen_list


def chain_list_tf_npiter(*iterables):
    """ iterables must be a list, tuple of preferable iterables or generators
    """
    for it in iterables:
        for element in it: 
            yield element


def dict_to_arr(d, bands):
    data, label = [], []
    for b in bands:
        if b == 'cwf':
            label.append(d[b])
        else:
            data.append(d[b])
    return (np.stack(data),np.stack(label))

def dict_gen_to_arr_arr(dict_gen, data_bands):
    data_li,label_li = [], []
    for img_dict in dict_gen:
        data,label = dict_to_arr(img_dict, data_bands)
        data_li.append(data)
        label_li.append(label)
    return (np.stack(data_li), np.stack(label_li))


def normalize_array(array,num_bands=24):
    # array in shape [N,num_bands,H,W]
    for ch in range(num_bands):
        # since arrays are float32,normlaize to range [0,1] for pytorch
        array[:,ch,:,:] = (array[:,ch,:,:]-array[:,ch,:,:].min())/(array[:,ch,:,:].max()-array[:,ch,:,:].min())    
    return array 
       

def exclude_isolated_pixels(arr, thr=2):
    from scipy import ndimage
    label_arr, num_labels = ndimage.label(arr)
    sizes = ndimage.sum_labels(arr, label_arr, range(num_labels + 1))    
    mask = sizes > thr # at least thr negihboring pixel
    new_arr = 1 * mask[label_arr]
    num_excluded_pixels = (arr ^ new_arr).sum()
    return new_arr, num_excluded_pixels


def select_on_classRatio_and_exclude_isolated_pixels(ds_np_gen , class_ratio=0.5, img_size = 256*256, exclude_thr=4):
    """ takes a numpy iterator of img_dic of tiles and selects only img tiles that have higher  class ratio than the threshold
    and removes pixel with less than thr negihbors
    """
    cls_thr = class_ratio * img_size
    for img_dic in  ds_np_gen:
        img = img_dic['cwf']
        if np.count_nonzero(img) > cls_thr:
            img_dic['cwf'], _ = exclude_isolated_pixels(img, exclude_thr)
            yield img_dic

def remove_mismatched_data_label_tiles(data, label, nuniq_thr=709):
    """removes erroneous tiles that happen in consecutive indices when exporting tiles from GEE due to boundary
       nuniq was obtained by looking at histogram, see sanity_check_images notebook
    """  
    nuniq = np.array([np.unique(img).size for img in data[:,[2,1,0],:,:] ])
    err_idxs = np.where(nuniq < nuniq_thr)[0] 
    idxs_to_choose = list(set(np.arange(data.shape[0]))-set(err_idxs))
    return data[idxs_to_choose], label[idxs_to_choose]


def get_fp_mask(gt,pred):
    """inputs: ground truth and pred (hard) mask numpy arrays and outputs the fp mask""" 
    ix, iy = np.where(pred > 0)
    fp = np.zeros(gt.shape)
    for x,y in zip(ix,iy):
        fp[x,y] = pred[x,y]-gt[x,y]
        
    return fp

def get_fn_mask(gt,pred):
    """inputs: ground truth and pred (hard) mask numpy arrays and outputs the fn mask"""
    ix, iy = np.where(gt > 0)
    fn = np.zeros(gt.shape)
    for x,y in zip(ix,iy):
        fn[x,y] = gt[x,y]-pred[x,y]

    return fn
