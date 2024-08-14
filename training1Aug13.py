# import time
# import torch
# import pytorch_lightning as pl
# import numpy as np
# import os




# class ModelLightningModule(pl.LightningModule):
#     def __init__(self, model, loss_fn, lr, model_name, baseline_loss=float('inf')):
#         super(ModelLightningModule, self).__init__()
#         self.model = model
#         self.loss_fn = loss_fn
#         self.lr = lr
#         self.model_name = model_name
#         self.total_train_loss_epoch = []
#         self.single_losses_epoch = {}
#         self.epoch_start_time = None
#         self.epoch_end_time = None
#         self.total_training_time = []
#         self.epoch_training_time = [] 
#         self.single_losses_gradient_epoch ={}
#         self.automatic_optimization = False  # Disable automatic optimization 
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#     def forward(self, x):
#         return self.model(x)

#     def on_train_epoch_start(self):
#         self.epoch_start_time = time.time()

#     def training_step(self, batch, batch_idx):
#         self.optimizer.zero_grad()  # Clear gradients at the start of each epoch   
#         model_input, gt = batch
#         model_output = self(model_input)
#         losses = self.loss_fn(model_output, gt)
#         total_loss = sum(loss.mean() for loss in losses.values())       
#         # Compute gradients for each loss component
#         for name, loss in losses.items():
#             self.manual_backward(loss.mean(),retain_graph=True)
#             grad_dict = {}
#             for param_name, param in self.model.named_parameters():
#                 if param.grad is not None:
#                     grad_dict[param_name] = param.grad.abs().detach().cpu().numpy().flatten()
#             self.single_losses_gradient_epoch[name] = grad_dict
#             if name not in self.single_losses_epoch:
#                 self.single_losses_epoch[name] = []
#                 self.single_losses_epoch[name].append(loss.mean().item())
#                 self.log(f'{name}_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#             # Log gradients
#             for param_name, grad_value in grad_dict.items():
#                 self.log(f'{name}_grad_{param_name}', grad_value, on_epoch=True, prog_bar=False, logger=True)
         
#             # Clear gradients for the next loss component
#             self.optimizer.zero_grad()
        
#         self.optimizer.zero_grad()
#         # Compute gradients for the total loss (this will be used for the actual update)
#         self.manual_backward(total_loss)
        
#         # Perform the optimization step
#         self.optimizer.step()
        
#         self.total_train_loss_epoch.append(total_loss.item())
#         return total_loss

#     def save_checkpoint(self, filename):
#         checkpoint_path = os.path.join(self.logger.log_dir, 'checkpoints', filename)
#         os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
#         torch.save(self.state_dict(), checkpoint_path)
    

#     def on_train_epoch_end(self):
#         self.epoch_end_time = time.time()
#         epoch_training_time = self.epoch_end_time - self.epoch_start_time
#         self.epoch_training_time.append(epoch_training_time)
#         total_time = sum(self.epoch_training_time)
#         self.total_training_time.append(total_time)  
#         self.log('train_loss_epoch',self.total_train_loss_epoch[-1], on_epoch=True, prog_bar=True,logger=True)
#         self.log('epoch_training_time', epoch_training_time, on_epoch=True, prog_bar=True, logger=True)
#         self.log('total_training_time', total_time, on_epoch=True, prog_bar=True,logger=True)
#         if self.current_epoch % 1000 == 0: 
#             checkpoint_filename = f"{self.model_name}_epoch_{self.current_epoch}.ckpt"
#             self.save_checkpoint(checkpoint_filename)
#             #self.plot_value_function(self.current_epoch) 

#     def on_train_end(self):
        
#         final_checkpoint = f"{self.model_name}_final.ckpt"
#         torch.save(self.state_dict(), final_checkpoint)
#         final_epoch_training_loss_text = f'Model: {self.model_name}, Final Epoch Training Loss: {self.total_train_loss_epoch[-1]}'
#         final_total_training_time_text = f'Model: {self.model_name}, Final Total Training Time: {self.total_training_time[-1]}'
#         self.logger.experiment.add_text('Final Epoch Training Loss', final_epoch_training_loss_text, self.current_epoch)
#         self.logger.experiment.add_text('Final Total Training Time', final_total_training_time_text, self.current_epoch)
#         for name, values in self.single_losses_epoch.items():
#             final_loss = values[-1]
#             final_loss_text = f'Model: {self.model_name}, Final {name} Loss: {final_loss}'
#             self.logger.experiment.add_text(f'Final {name} Loss', final_loss_text, self.current_epoch)   


#     @classmethod
#     def load_from_checkpoint(cls, checkpoint_path, model, loss_fn, lr, model_name):
#         checkpoint = torch.load(checkpoint_path)
#         instance = cls(model, loss_fn, lr, model_name)
#         instance.load_state_dict(checkpoint)
#         return instance        
    


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



    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()  # Clear gradients at the start of each epoch   
        model_input, gt = batch
        model_output = self(model_input)
        losses = self.loss_fn(model_output, gt)
        total_loss = sum(loss.mean() for loss in losses.values())

        all_gradients = {name: [] for name in losses.keys()}
        
        # Compute gradients for each loss component
        for name, loss in losses.items():
            self.manual_backward(loss.mean(), retain_graph=True)
            grad_dict = {}
            for param_name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_values = param.grad.abs().detach().cpu().numpy().flatten()
                    grad_dict[param_name] = grad_values
                    all_gradients[name].extend(grad_values)
            self.single_losses_gradient_epoch[name] = grad_dict
            if name not in self.single_losses_epoch:
                self.single_losses_epoch[name] = []
            self.single_losses_epoch[name].append(loss.mean().item())
            self.log(f'{name}_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            # Log gradients
            for param_name, grad_value in grad_dict.items():
                self.log(f'{name}_grad_{param_name}', np.mean(grad_value), on_epoch=True, prog_bar=False, logger=True)
            
            # Clear gradients for the next loss component
            self.optimizer.zero_grad()

        for name, grads in all_gradients.items():
            max_grad = np.max(grads)
            mean_abs_grad = np.mean(np.abs(grads))
            if name not in self.max_grad_components:
                self.max_grad_components[name] = []
            self.max_grad_components[name].append(max_grad)
            self.log(f'{name}_max_grad', max_grad, on_epoch=True, prog_bar=True, logger=True)
            if name not in self.mean_abs_grads:
                self.mean_abs_grads[name] = []
            self.mean_abs_grads[name].append(mean_abs_grad)
            self.log(f'{name}_mean_abs_grad', mean_abs_grad, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{name}_max_grad', max_grad, on_epoch=True, prog_bar=True, logger=True)
           

        
        self.optimizer.zero_grad()
        # Compute gradients for the total loss (this will be used for the actual update)
        self.manual_backward(total_loss)
        
        # Perform the optimization step
        self.optimizer.step()
        
        self.total_train_loss_epoch.append(total_loss.item())

        # Store gradients for histogram plotting
        if self.current_epoch % self.histogram_log_frequency == 0:
            self.gradient_histograms[self.current_epoch] = all_gradients

        return total_loss

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
            data[f'{name}_max_grad'] = self.max_grad_components[name]
            data[f'{name}_mean_abs_grad'] = self.mean_abs_grads[name]
    
        df = pd.DataFrame(data)
        df.to_csv(f'{self.model_name}_gradient_stats.csv', index=False)


    def on_train_end(self):
        final_checkpoint = f"{self.model_name}_final.ckpt"
        torch.save(self.state_dict(), final_checkpoint)
        final_epoch_training_loss_text = f'Model: {self.model_name}, Final Epoch Training Loss: {self.total_train_loss_epoch[-1]}'
        final_total_training_time_text = f'Model: {self.model_name}, Final Total Training Time: {self.total_training_time[-1]}'
        self.logger.experiment.add_text('Final Epoch Training Loss', final_epoch_training_loss_text, self.current_epoch)
        self.logger.experiment.add_text('Final Total Training Time', final_total_training_time_text, self.current_epoch)
        for name, values in self.single_losses_epoch.items():
            final_loss = values[-1]
            final_loss_text = f'Model: {self.model_name}, Final {name} Loss: {final_loss}'
            self.logger.experiment.add_text(f'Final {name} Loss', final_loss_text, self.current_epoch)
        for name, max_grads in self.max_grad_components.items():
            final_max_grad = max_grads[-1]
            final_max_grad_text = f'Model: {self.model_name}, Final {name} Max Gradient: {final_max_grad}'
            self.logger.experiment.add_text(f'Final {name} Max Gradient', final_max_grad_text, self.current_epoch)
        self.save_gradient_stats_to_csv()    
    

    def plot_gradient_histograms(self, epoch):
        gradients = self.gradient_histograms[epoch]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, grads in gradients.items():
            ax.hist(grads, bins=50, alpha=0.5, label=name)
        
        ax.set_yscale('log')
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Count')
        ax.set_title(f'Gradient Histograms at Epoch {epoch}')
        ax.legend()
        
        # Save or log the figure using your preferred method
        # For example, if using TensorBoard:
        self.logger.experiment.add_figure(f'Gradient_Histograms_Epoch_{epoch}', fig, epoch)
        plt.close(fig)  # Close the figure to free up memory

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model, loss_fn, lr, model_name):
        checkpoint = torch.load(checkpoint_path)
        instance = cls(model, loss_fn, lr, model_name)
        instance.load_state_dict(checkpoint)
        return instance