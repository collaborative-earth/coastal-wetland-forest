#pylint: disable-all
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
import lightning as L
from torchmetrics.classification import BinaryCohenKappa, BinaryF1Score
from torchvision.ops import sigmoid_focal_loss
from torch.autograd import Variable

class CFWModel(L.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, augmentation = True,
                  resize_crop=(128,128), learning_rate=0.001, weight_decay = 0.7, **kwargs):
        super().__init__()
        self.model = smp.create_model( arch, encoder_name=encoder_name, encoder_weights=None, 
                                      in_channels=in_channels, classes=out_classes,  **kwargs )      
        self.save_hyperparameters()
        self.augmentation = augmentation
        self.transform = {"train": transforms.Compose([transforms.RandomResizedCrop(resize_crop)]),
                          "valid" : transforms.Compose([transforms.Resize(resize_crop)]),
                          "test" : transforms.Compose([transforms.Resize(resize_crop)])
                         }
                          
        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.lr = learning_rate
        self.wd = weight_decay
        
    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch[0]
       
        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4
        
        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        mask = batch[1]

        if self.augmentation:
            image = self.transform[stage](image) 
            mask = self.transform[stage](mask)
 

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        #loss  = sigmoid_focal_loss(logits_mask, mask,reduction='mean',gamma=5)
        

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
       
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary", threshold=None)
        metric = BinaryCohenKappa().to(self.device)
        kappa = metric( pred_mask.long(), mask.long())
        f1_metric = BinaryF1Score().to(self.device)
        f1 = f1_metric(pred_mask.long(),mask.long())
        
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "kappa": kappa, 
            "f1": f1, 
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"].float() for x in outputs])
        fp = torch.cat([x["fp"].float() for x in outputs])
        fn = torch.cat([x["fn"].float() for x in outputs])
        tn = torch.cat([x["tn"].float() for x in outputs])
        kappa  = torch.stack([x["kappa"] for x in outputs])
        f1  = torch.stack([x["f1"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs])

        #per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_accuracy = smp.metrics.accuracy(tp,fp,fn,tn,reduction="micro-imagewise") 
        #dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        cohenKappa = torch.mean(kappa)
        f1score = torch.mean(f1)
        loss = torch.mean(loss)
        tp = torch.sum(tp)
        fn = torch.sum(fn)
        tn = torch.sum(tn)
        fp = torch.sum(fp)
        tpr = tp/(tp+fn)
        fpr = 1-(tn/(tn+fp))
        fnr = 1-tpr
        

        metrics = {
            #f"{stage}_per_image_iou": per_image_iou,
            #f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_kappa": cohenKappa,
            f"{stage}_acc": per_image_accuracy,
            #f"{stage}_accuracy": (tp+tn) / (tp+tn+fp+fn),
            f"{stage}_loss":loss,
            f"{stage}_fnr": fnr,
            f"{stage}_fpr": fpr,
            f"{stage}_f1": f1score,
            'step': self.current_epoch
        }
       
        self.log_dict(metrics, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.8)
        return [optimizer], [scheduler]
     