# pylint: disable-all 
import json
import subprocess
from google.cloud import storage
import tensorflow as tf

class GoogleStorageReader:
    """read files from google storage bucket and separate them into tfrecord, jason, file lists
    """
    def __init__(self, bucket_name):
        self._bucket_name = bucket_name
        self._files = []
        self._tfr_files = []
        self._mixer = None

    def list_files(self):
        """ returns a list (full path) in the bucket 
        raises an exception if no files is found
        """
        cmd  = f"gsutil ls gs://{self._bucket_name}"
        status,files = subprocess.getstatusoutput(cmd)
        if status == 0:
            self._files = files.split("\n")
            self.list_tfrecord_files()
            self.get_json_file()
            return self._files
        else:
            raise FileNotFoundError("No file was found")

    def list_tfrecord_files(self, suffix='tfrecord'):
        """ returns a list (full path) of tfrecord files in the bucket
        """
        self._tfr_files = [ f for f in self._files if f.endswith(suffix) or f.endswith('gz')]
        return self._tfr_files

    def get_json_file(self,file_prefix="mixer.json"):
        """returns the json mixer file in the bucket
        raises an exception if no files is found
        """
        json_file_list = [f for f in self._files if file_prefix in f]
        if json_file_list:
            json_file = json_file_list[0] #str(path/(file_prefix+'mixer.json'))
            cmd = f"gsutil cat {json_file}"
            status,text = subprocess.getstatusoutput(cmd)
            if status == 0:
                self._mixer = json.loads(text)
                return self._mixer
            raise FileNotFoundError("No json file was found")

        
    def get_patch_dict(self):
        """returns patch(tile) information from mixer json data
        """
        if self._mixer:
            patch_dict = dict(
            width = self._mixer['patchDimensions'][0],
            height = self._mixer['patchDimensions'][1],
            patchesTotal = self._mixer['totalPatches'],
            dimensions_flat = [ self._mixer['patchDimensions'][0],  self._mixer['patchDimensions'][1] ],
            size = self._mixer['patchDimensions'][0] * self._mixer['patchDimensions'][1] 
            )
            return patch_dict
        else:
            return None



class Tile:
    """stores properties in the format need of an tfr image tile object"""
    def __init__(self, bucket_name, data_bands=['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','NDVI','NDWI','SR',
                                     'EVI','OSAVI','SR_B2_1','SR_B3_1','SR_B4_1','SR_B5_1','SR_B6_1','SR_B7_1',
                                     'ST_B10_1','NDVI_1','NDWI_1','SR_1','EVI_1','OSAVI_1'],
                                       label_name='cwf' ):
        self._data_bands = data_bands
        self._label_name = label_name
        self._bucket_name = bucket_name
        
    @property
    def label_name(self):
        """ name of label (cwf)"""
        return self._label_name
    
    @property
    def data_bands(self):
        """name of image data bands"""
        return self._data_bands
    
    def data_label_bands(self):
        """ adds label band to data bands"""
        return self.data_bands + [self.label_name]
    
    def _tf_feature_dict(self):
        """creates image feature dictionary for parsing tfr image"""
        gs_reader = GoogleStorageReader(self._bucket_name)
        gs_reader.list_files()
        dimensions_flat = gs_reader.get_patch_dict()['dimensions_flat']   
        image_columns = [tf.io.FixedLenFeature(shape=dimensions_flat, dtype=tf.float32) for k in self.data_bands]
        image_columns += [tf.io.FixedLenFeature(shape=dimensions_flat, dtype=tf.int64)]
        return dict(zip(self.data_label_bands(), image_columns))
    
    def feature_dict(self):
        """returns featuer dictionary"""
        return self._tf_feature_dict()   
    

# bucket_name = "image_tiles_us_florida"       
# gs_handler = GoogleStorageReader(bucket_name) 
# files = gs_handler.list_files()  
# tfr_files_list = gs_handler.list_tfrecord_files()
# mixer = gs_handler.get_json_file()
#print(gs_handler.get_patch_dict())

# tfrm = Tile(bucket_name)
# print(tfrm.data_bands)
# print(tfrm.label_name)
# print(tfrm.data_label_bands())
# print(tfrm.feature_dict())
