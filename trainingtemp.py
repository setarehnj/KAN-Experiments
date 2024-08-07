import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os


class ModelLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, model_name, baseline_loss=float('inf')):
        super(ModelLightningModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.model_name = model_name
        self.baseline_loss = baseline_loss
        self.single_losses_epoch = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.epoch_start_time = None
        self.epoch_end_time = None
        self.total_training_times = []
        self.iteration_times = []

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()



    def training_step(self, batch, batch_idx):
        model_input, gt = batch
        model_output = self(model_input)
        losses = self.loss_fn(model_output, gt)
        single_loss = sum(loss.mean() for loss in losses.values())
        self.single_losses_epoch.append(single_loss.item())
        return single_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.logger.log_dir, 'checkpoints', filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)
    

    def on_train_epoch_end(self):
        self.epoch_end_time = time.time()
        epoch_training_time = self.epoch_end_time - self.epoch_start_time
        self.iteration_times.append(epoch_training_time)
        total_time = sum(self.iteration_times)
        self.total_training_times.append(total_time) 
        self.log('train_loss_epoch', single_losses_epoch[-1], on_epoch=True, prog_bar=True, sync_dist= True, logger=True)
        self.log('epoch_training_time', epoch_training_time, on_epoch=True, prog_bar=True, sync_dist = True, logger=True)
        self.log('total_training_time', total_time, on_epoch=True, prog_bar=True, sync_dist= True, logger=True)
        if self.current_epoch % 1000 == 0: 
            checkpoint_filename = f"{self.model_name}_epoch_{self.current_epoch}.ckpt"
            self.save_checkpoint(checkpoint_filename)

    
    def on_train_end(self):
        
        final_checkpoint = f"{self.model_name}_final.ckpt"
        torch.save(self.state_dict(), final_checkpoint)
        final_epoch_training_loss_text = f'Model: {self.model_name}, Final Epoch Training Loss: {self.single_losses_epoch[-1]}'
        final_total_training_time_text = f'Model: {self.model_name}, Final Total Training Time: {self.total_training_times[-1]}'
        self.logger.experiment.add_text('Final Epoch Training Loss', final_epoch_training_loss_text, self.current_epoch)
        self.logger.experiment.add_text('Final Total Training Time', final_total_training_time_text, self.current_epoch)    
