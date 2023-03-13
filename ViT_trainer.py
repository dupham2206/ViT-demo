import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from vision_transformer import VisionTransformer
import wandb

class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        wandb.init(project="ViT-demo", name="ViT 16 layers MNIST")

    # inference only
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]
    
    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        if(mode == "train"):
            self.log("%s_loss" % mode, loss.item(), prog_bar=True)
            self.log("%s_acc" % mode, acc.item(), prog_bar=True)

        if(mode == "val"):
            wandb.log({
                "val_loss": loss,
                "val_acc": acc, 
            })
        return loss 

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode = "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode = "val")
    
    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode = "test")

