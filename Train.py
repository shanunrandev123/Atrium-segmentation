#%%
# from model import UNet
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytor
# import numpy as np
# from dataset import CardiacDataset
# import matplotlib.pyplot as plt
# import imgaug.augmenters as iaa
# from pathlib import Path
# from torch.utils.data import DataLoader


#%%
from model import UNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint




#%%

seq = iaa.Sequential([iaa.Affine(scale=(0.85, 1.15), rotate=(-45,45)),iaa.ElasticTransformation()])
#%%
train_path = Path("Preprocessed/train/")

val_path = Path("Preprocessed/val/")
#%%
train_dataset = CardiacDataset(train_path, seq)

val_dataset = CardiacDataset(val_path, None)
#%%
batch_size = 8

num_workers = 4

# train_loader = torch.utils.DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle = True)
# val_loader = torch.utils.DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, shuffle = False)


train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size, num_workers=num_workers, shuffle=False)


# %%

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, mask):
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        
        counter = (pred * mask).sum()
        denum = pred.sum() + mask.sum() + 1e-7
        dice = (2*counter) / denum
        return 1 - dice
    
# %%


class AtriumSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.loss_fn = DiceLoss()
        
    
    def forwward(self, data):
        return torch.sigmoid(self.model(data))
    
    
    def training_step(self, batch):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)
        
        loss = self.loss_fn(pred, mask)
        
        self.log("Train Dice", loss)
        
        if batch_idx % 50 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Train")
            
        return loss
    
    
    
    def validation_step(self, batch):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)
        
        loss = self.loss_fn(pred, mask)
        
        self.log("val Dice", loss)
        
        if batch_idx % 2 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Val")
            
        return loss
    
    
    def log_images(self, mri, pred, mask, name):
        pred = pred > 0.5
        fig, axis = plt.subplots(1,2)
        axis[0].imshow(mri[0][0], cmap = 'bone')
        mask = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
        axis[0].imshow(mask, alpha = 0.6)
        
        axis[1].imshow(mri[0][0], cmap = 'bone')
        mask = np.ma.masked_where(pred[0][0] == 0, pred[0][0])
        axis[1].imshow(mask, alpha = 0.6)
        
        self.logger.experiment.add_figure(name, fig, self.global_step)
        
        
    
    
    def configure_optim(self):
        return [self.optimizer]
    
    
    
    
    
    
    
# %%
torch.manual_seed(0)

model = AtriumSegmentation()
# %%
checkpoint_callback = ModelCheckpoint(monitor="Val Dice", save_top_k=10, mode='min')

trainer = pl.Trainer(logger=TensorBoardLogger(save_dir='logs'), log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=15)


# %%
