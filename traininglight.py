import torch
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
    convergence_dir = os.path.join(root_path, 'convergence_curves')
    
    if not os.path.exists(convergence_dir):
        os.makedirs(convergence_dir)
    
    plt.figure()
    plt.plot(kan_losses, label='Chebyshev KAN')
    plt.plot(mlp_losses, label='MLP')
    plt.plot(fourier_losses, label='Fourier KAN')
    plt.plot(wavelet_losses, label='Wavelet KAN')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Convergence Curves')
    plt.legend()
    if epoch is not None:
        plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
    else:
        plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
    plt.close()

class ModelLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, steps_til_summary, validation_fn=None):
        super(ModelLightningModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.steps_til_summary = steps_til_summary
        self.validation_fn = validation_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        model_input, gt = batch
        model_output = self(model_input)
        losses = self.loss_fn(model_output, gt)
        train_loss = sum([loss.mean() for loss in losses.values()])
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        if self.validation_fn is not None:
            self.validation_fn(self.model, batch_idx)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def on_epoch_end(self):
        if self.current_epoch % self.steps_til_summary == 0:
            torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_epoch_{self.current_epoch}.ckpt")

class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)