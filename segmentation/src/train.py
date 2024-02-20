#pylint: disable-all
#TODO:
# curretly this code sementation fautl error while the notebook train.pynb works!!
import tensorflow as tf
import os
import sys
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

module_path = os.path.abspath(".")
if module_path not in sys.path: sys.path.append(module_path)
from src.data import GoogleStorageReader, Tile
from src.utils import chain_list_tf_npiter,dict_gen_to_arr_arr, normalize_array, list_gen
from src.model import CFWModel

import wandb
wandb.login()
#-----
#Parameters
bucket_name = "gee_image_us_landsat8_ccap_band13_tile_size128"
bucket_name_for_processed_tiles = "processed_tiles"
batch_size = 8
num_in_channel = 24
max_epochs = 1
arch = "UNet"
encoder = "inceptionresnetv2"
eval_on_testset = False
#------

# Disable all GPUS
try: 
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass
    #raise Exception("Invalid device or cannot disable GPU")

#----
#get data from google storage bucke
gs_handler = GoogleStorageReader(bucket_name) 
files = gs_handler.list_files()  
tfr_files_list = gs_handler.list_tfrecord_files()
mixer = gs_handler.get_json_file()
tile = Tile(bucket_name)
image_features_dict = tile.feature_dict()
processed_gs_handler = GoogleStorageReader(bucket_name_for_processed_tiles) 
processed_files = processed_gs_handler.list_files()  
processed_tfr_files_list = processed_gs_handler.list_tfrecord_files()
gen_list = list_gen(processed_tfr_files_list,len(tfr_files_list),image_features_dict)
data_bands = tile.data_label_bands()
gen_data = chain_list_tf_npiter(*gen_list)
train_data, label_data = dict_gen_to_arr_arr(gen_data,data_bands)
norm_data = normalize_array(train_data)
del train_data

#-------
#split data
x_, x_test, y_, y_test = train_test_split(norm_data, label_data,
                                                test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_,y_, 
                                                test_size=0.2, random_state=42)
del norm_data

xtrain = torch.Tensor(x_train) 
ytrain = torch.Tensor(y_train)
xval = torch.Tensor(x_val) 
yval = torch.Tensor(y_val)
xtest = torch.Tensor(x_test) 
ytest = torch.Tensor(y_test)

train_dataset = TensorDataset(xtrain,ytrain) 
val_dataset = TensorDataset(xval,yval) 
test_dataset = TensorDataset(xtest,ytest) 

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=n_cpu) 
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True, num_workers=n_cpu) 
test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle= False, num_workers=n_cpu) 

wandb_logger = WandbLogger(project='cfw_segmentation',name='ep100_7re-06_wd.3')

# define model
model = CFWModel(arch, encoder, in_channels=num_in_channel, augmentation = False, out_classes=1)

early_stop = EarlyStopping(monitor="valid_kappa", patience=20, mode="max")
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid_kappa", mode="max")

#train
trainer = pl.Trainer(accelerator="cpu", logger=wandb_logger, max_epochs= max_epochs,
                     callbacks=[RichProgressBar(), early_stop, checkpoint_callback],
                     auto_lr_find=True)

# find the starting lr, and plot the loss-vs-lr
# result = trainer.tune(model,train_dataloaders=train_dataloader)
# result['lr_find'].plot(suggest=True);

print("start training")

trainer.fit( model, train_dataloaders=train_dataloader,  val_dataloaders=val_dataloader)

print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
print(checkpoint_callback.best_model_score.item()) # and prints it score


# evaluate on test dataset
if eval_on_testset:
    print("evaluating performance on the testset")
    trainer.test(model=best_model, dataloaders=test_dataloader, ckpt_path='best')

wandb.finish()
