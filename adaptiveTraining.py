import time
import torch
import pytorch_lightning as pl
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

class ModelLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, model_name, histogram_log_frequency=1000):
        super(ModelLightningModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.model_name = model_name
        self.total_train_loss_epoch = []
        self.single_losses_epoch = {}
        self.epoch_start_time = None
        self.epoch_end_time = None
        self.total_training_time = []
        self.epoch_training_time = []
        self.single_losses_gradient_epoch = {}
        self.automatic_optimization = False  # Disable automatic optimization 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.gradient_histograms = {}
        self.histogram_log_frequency = histogram_log_frequency
        self.max_grad_components = {}
        self.mean_abs_grads = {}  # to store mean absolute gradients for each loss
        self.var_grads = {}  # Variances of gradients
        self.weights = {}  # Adaptive weights
        self.alpha = 0.9  # Alpha parameter for weight update

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()  # Clear gradients at the start of each epoch   
        model_input, gt = batch
        model_output = self(model_input)
        losses = self.loss_fn(model_output, gt)
        total_loss = 0

        all_gradients = {name: [] for name in losses.keys()}
        var_grads = {}  # To store variance of gradients for this step
        
        # Compute gradients for each loss component
        for name, loss in losses.items():
            loss_mean = loss.mean()
            if loss_mean.item() != 0 and loss_mean.requires_grad:
                self.manual_backward(loss_mean, retain_graph=True)
                
                grad_dict = {}
                grads_flatten = []
                for param_name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_values = param.grad.detach().cpu().numpy().flatten()
                        grads_flatten.extend(grad_values)
                        grad_dict[param_name] = grad_values
                        all_gradients[name].extend(grad_values)

                if grads_flatten:
                    var_grads[name] = np.var(grads_flatten)
                else:
                    var_grads[name] = 0

                # Log gradients
                for param_name, grad_value in grad_dict.items():
                    self.log(f'{name}_grad_{param_name}', np.mean(grad_value), on_epoch=True, prog_bar=False, logger=True)
                
                if name not in self.single_losses_epoch:
                    self.single_losses_epoch[name] = []
                self.single_losses_epoch[name].append(loss_mean.item())
                self.log(f'{name}_loss', loss, on_epoch=True, prog_bar=True, logger=True)
                
                # Clear gradients for the next loss component
                self.optimizer.zero_grad()

        # Update weights using the provided formula
        for name in losses.keys():
            if name not in self.weights:
                self.weights[name] = 1.0
            if var_grads[name] != 0:
                max_var = max(var_grads.values())
                self.weights[name] = (max_var / var_grads[name]) * (1 - self.alpha) + self.alpha * self.weights.get(name, 1.0)

        # Compute total loss using adaptive weights
        for name, loss in losses.items():
            total_loss += self.weights[name] * loss.mean()
        
        self.optimizer.zero_grad()
        self.manual_backward(total_loss)
        self.optimizer.step()
        
        self.total_train_loss_epoch.append(total_loss.item())

        # Store gradients for histogram plotting
        if self.current_epoch % self.histogram_log_frequency == 0:
            self.gradient_histograms[self.current_epoch] = all_gradients

        # Log adaptive weights
        for name, weight in self.weights.items():
            self.log(f'{name}_weight', weight, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.optimizer

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.logger.log_dir, 'checkpoints', filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def on_train_epoch_end(self):
        self.epoch_end_time = time.time()
        epoch_training_time = self.epoch_end_time - self.epoch_start_time
        self.epoch_training_time.append(epoch_training_time)
        total_time = sum(self.epoch_training_time)
        self.total_training_time.append(total_time)  
        self.log('train_loss_epoch', self.total_train_loss_epoch[-1], on_epoch=True, prog_bar=True, logger=True)
        self.log('epoch_training_time', epoch_training_time, on_epoch=True, prog_bar=True, logger=True)
        self.log('total_training_time', total_time, on_epoch=True, prog_bar=True, logger=True)
        
        if self.current_epoch % 1000 == 0: 
            checkpoint_filename = f"{self.model_name}_epoch_{self.current_epoch}.ckpt"
            self.save_checkpoint(checkpoint_filename)
        
        if self.current_epoch in self.gradient_histograms:
            self.plot_gradient_histograms(self.current_epoch)

    def save_gradient_stats_to_csv(self):
        data = {
            'epoch': range(len(self.total_train_loss_epoch)),
            'total_loss': self.total_train_loss_epoch
        }
    
        for name in self.single_losses_epoch.keys():
            data[f'{name}_loss'] = self.single_losses_epoch[name]
            data[f'{name}_max_grad'] = s
