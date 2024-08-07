# # import torch
# # import pytorch_lightning as pl
# # from torch.utils.tensorboard import SummaryWriter
# # from tqdm.autonotebook import tqdm
# # import time
# # import numpy as np
# # import os
# # import shutil
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt

# # def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
# #     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
# #     if not os.path.exists(convergence_dir):
# #         os.makedirs(convergence_dir)
    
# #     plt.figure()
# #     plt.plot(kan_losses, label='Chebyshev KAN')
# #     plt.plot(mlp_losses, label='MLP')
# #     plt.plot(fourier_losses, label='Fourier KAN')
# #     plt.plot(wavelet_losses, label='Wavelet KAN')
# #     plt.yscale('log')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.title('Convergence Curves')
# #     plt.legend()
# #     if epoch is not None:
# #         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
# #     else:
# #         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
# #     plt.close()

# # class ModelLightningModule(pl.LightningModule):
# #     def __init__(self, model, loss_fn, lr, steps_til_summary, validation_fn=None):
# #         super(ModelLightningModule, self).__init__()
# #         self.model = model
# #         self.loss_fn = loss_fn
# #         self.lr = lr
# #         self.steps_til_summary = steps_til_summary
# #         self.validation_fn = validation_fn

# #     def forward(self, x):
# #         return self.model(x)

# #     def training_step(self, batch, batch_idx):
# #         model_input, gt = batch
# #         model_output = self(model_input)
# #         losses = self.loss_fn(model_output, gt)
# #         train_loss = sum([loss.mean() for loss in losses.values()])
# #         self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
# #         return train_loss

# #     def validation_step(self, batch, batch_idx):
# #         if self.validation_fn is not None:
# #             self.validation_fn(self.model, batch_idx)
        
# #     def configure_optimizers(self):
# #         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
# #         return optimizer

# #     def on_epoch_end(self):
# #         if self.current_epoch % self.steps_til_summary == 0:
# #             torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_epoch_{self.current_epoch}.ckpt")

# # class LinearDecaySchedule():
# #     def __init__(self, start_val, final_val, num_steps):
# #         self.start_val = start_val
# #         self.final_val = final_val
# #         self.num_steps = num_steps

# #     def __call__(self, iter):
# #         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

# # import torch
# # import pytorch_lightning as pl
# # from torch.utils.tensorboard import SummaryWriter
# # from tqdm.autonotebook import tqdm
# # import time
# # import numpy as np
# # import os
# # import shutil
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import pandas as pd

# # # Check CUDA availability and GPU devices
# # print("CUDA available:", torch.cuda.is_available())  # Should return True
# # print("Number of GPUs:", torch.cuda.device_count())  # Should return 2
# # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
# #     print("GPU 0:", torch.cuda.get_device_name(0))  # Should return the name of the first GPU
# #     print("GPU 1:", torch.cuda.get_device_name(1))  # Should return the name of the second GPU


# # def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
# #     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
# #     if not os.path.exists(convergence_dir):
# #         os.makedirs(convergence_dir)
    
# #     plt.figure()
# #     plt.plot(kan_losses, label='Chebyshev KAN')
# #     plt.plot(mlp_losses, label='MLP')
# #     plt.plot(fourier_losses, label='Fourier KAN')
# #     plt.plot(wavelet_losses, label='Wavelet KAN')
# #     plt.yscale('log')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.title('Convergence Curves')
# #     plt.legend()
# #     if epoch is not None:
# #         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
# #     else:
# #         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
# #     plt.close()

# # class ModelLightningModule(pl.LightningModule):
# #     def __init__(self, model, loss_fn, lr, steps_til_summary, validation_fn=None):
# #         super(ModelLightningModule, self).__init__()
# #         self.model = model
# #         self.loss_fn = loss_fn
# #         self.lr = lr
# #         self.steps_til_summary = steps_til_summary
# #         self.validation_fn = validation_fn
# #         self.iteration_times = []
# #         self.total_training_times = []
# #         self.losses = []

# #     def forward(self, x):
# #         return self.model(x)

# #     def training_step(self, batch, batch_idx):
# #         start_time = time.time()

# #         model_input, gt = batch
# #         model_output = self(model_input)
# #         losses = self.loss_fn(model_output, gt)
# #         train_loss = sum([loss.mean() for loss in losses.values()])
# #         self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

# #         iteration_time = time.time() - start_time
# #         self.iteration_times.append(iteration_time)
# #         self.total_training_times.append(sum(self.iteration_times))
        
# #         self.log('iteration_time', iteration_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
# #         self.log('total_training_time', self.total_training_times[-1], on_step=True, on_epoch=True, prog_bar=True, logger=True)

# #         self.losses.append(train_loss.item())
# #         return train_loss

# #     def validation_step(self, batch, batch_idx):
# #         if self.validation_fn is not None:
# #             self.validation_fn(self.model, batch_idx)

# #     def configure_optimizers(self):
# #         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
# #         return optimizer

# #     def on_epoch_end(self):
# #         if self.current_epoch % self.steps_til_summary == 0:
# #             torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_epoch_{self.current_epoch}.ckpt")

# # class LinearDecaySchedule():
# #     def __init__(self, start_val, final_val, num_steps):
# #         self.start_val = start_val
# #         self.final_val = final_val
# #         self.num_steps = num_steps

# #     def __call__(self, iter):
# #         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

# # import torch
# # import pytorch_lightning as pl
# # from torch.utils.tensorboard import SummaryWriter
# # from tqdm.autonotebook import tqdm
# # import time
# # import numpy as np
# # import os
# # import shutil
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import pandas as pd  # Ensure pandas is imported

# # # Check CUDA availability and GPU devices
# # print("CUDA available:", torch.cuda.is_available())  # Should return True
# # print("Number of GPUs:", torch.cuda.device_count())  # Should return 2
# # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
# #     print("GPU 0:", torch.cuda.get_device_name(0))  # Should return the name of the first GPU
# #     print("GPU 1:", torch.cuda.get_device_name(1))  # Should return the name of the second GPU

# # def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
# #     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
# #     if not os.path.exists(convergence_dir):
# #         os.makedirs(convergence_dir)
    
# #     plt.figure()
# #     plt.plot(kan_losses, label='Chebyshev KAN')
# #     plt.plot(mlp_losses, label='MLP')
# #     plt.plot(fourier_losses, label='Fourier KAN')
# #     plt.plot(wavelet_losses, label='Wavelet KAN')
# #     plt.yscale('log')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.title('Convergence Curves')
# #     plt.legend()
# #     if epoch is not None:
# #         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
# #     else:
# #         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
# #     plt.close()

# # class ModelLightningModule(pl.LightningModule):
# #     def __init__(self, model, loss_fn, lr, steps_til_summary, validation_fn=None):
# #         super(ModelLightningModule, self).__init__()
# #         self.model = model
# #         self.loss_fn = loss_fn
# #         self.lr = lr
# #         self.steps_til_summary = steps_til_summary
# #         self.validation_fn = validation_fn
# #         self.iteration_times = []
# #         self.total_training_times = []
# #         self.losses = []

# #     def forward(self, x):
# #         return self.model(x)

# #     def training_step(self, batch, batch_idx):
        

# #         model_input, gt = batch
# #         model_output = self(model_input)
# #         losses = self.loss_fn(model_output, gt)
# #         train_loss = sum([loss.mean() for loss in losses.values()])
# #         self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

# #         iteration_time = time.time() - start_time
# #         self.iteration_times.append(iteration_time)
# #         self.total_training_times.append(sum(self.iteration_times))
        
# #         self.log('iteration_time', iteration_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
# #         self.log('total_training_time', self.total_training_times[-1], on_step=True, on_epoch=True, prog_bar=True, logger=True)

# #         self.losses.append(train_loss.item())
# #         return train_loss

# #     def validation_step(self, batch, batch_idx):
# #         if self.validation_fn is not None:
# #             self.validation_fn(self.model, batch_idx)

# #     def configure_optimizers(self):
# #         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
# #         return optimizer

# #     def on_epoch_end(self):
# #         if self.current_epoch % self.steps_til_summary == 0:
# #             torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_epoch_{self.current_epoch}.ckpt")

# # class LinearDecaySchedule():
# #     def __init__(self, start_val, final_val, num_steps):
# #         self.start_val = start_val
# #         self.final_val = final_val
# #         self.num_steps = num_steps

# #     def __call__(self, iter):
# #         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

# import torch
# import pytorch_lightning as pl
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
# import time
# import numpy as np
# import os
# import shutil
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import pandas as pd  # Ensure pandas is imported
# from pytorch_lightning.loggers import TensorBoardLogger
# # def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
# #     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
# #     if not os.path.exists(convergence_dir):
# #         os.makedirs(convergence_dir)
    
# #     plt.figure()
# #     plt.plot(kan_losses, label='Chebyshev KAN')
# #     plt.plot(mlp_losses, label='MLP')
# #     plt.plot(fourier_losses, label='Fourier KAN')
# #     plt.plot(wavelet_losses, label='Wavelet KAN')
# #     plt.yscale('log')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.title('Convergence Curves')
# #     plt.legend()
# #     if epoch is not None:
# #         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
# #     else:
# #         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
# # #     plt.close()


# # class ModelLightningModule(pl.LightningModule):
# #     def __init__(self, model, loss_fn, lr, steps_til_summary, validation_fn=None):
# #         super(ModelLightningModule, self).__init__()
# #         self.model = model
# #         self.loss_fn = loss_fn
# #         self.lr = lr
# #         self.steps_til_summary = steps_til_summary
# #         self.validation_fn = validation_fn
# #         # self.iteration_times = []
# #         self.total_training_times = []
# #         self.losses = []
# #         self.core_iteration_time = None  # To track the core computation time at the end of each epoch
# #         self.train_loss_epoch = None  # To track the total training loss at the end of each epoch
# #         #self.save_hyperparameters(ignore=['model'])



# #     def forward(self, x):
# #         return self.model(x)

# #     # def on_after_backward(self):
# #     # # Log parameter histograms
# #     # for name, params in self.named_parameters():
# #     #     self.logger.experiment.add_histogram(name, params, self.current_epoch)
# #     #     if params.grad is not None:
# #     #         self.logger.experiment.add_histogram(f'{name}_grad', params.grad, self.current_epoch)    

# #     # def on_train_epoch_start(self):
# #     #     self.start_time = time.time()  # Capture the start time of the epoch

# #     def training_step(self, batch, batch_idx):

# #         model_input, gt = batch
# #         core_start_time = time.time()
# #         model_output = self(model_input)
# #         losses = self.loss_fn(model_output, gt)
# #         self.train_loss_epoch = sum([loss.mean() for loss in losses.values()])  # Calculate total loss
        
# #         # Measure core computation time (excluding logging)
# #         core_end_time = time.time()
# #         self.core_iteration_time = core_end_time - core_start_time
# #         self.total_training_times.append(self.core_iteration_time)
# #         # Log the training loss both at each step and at the end of the epoch
# #         # self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
# #         #self.log('iteration_time', self.core_iteration_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
# #         # Append the training loss for further analysis

# #         return self.train_loss_epoch

# #     def on_train_epoch_end(self):
         
# #         # Log iteration time and total training time at the end of each epoch
# #         self.losses.append(train_loss_epoch.item())

# #         self.log('train_loss', self.train_loss_epoch, on_epoch=True, prog_bar=True, logger=True)

# #         self.log('iteration_time', self.core_iteration_time, on_epoch=True, prog_bar=True, logger=True)

# #         # Save model checkpoint at specified intervals
# #         if self.current_epoch % self.steps_til_summary == 0:
# #             self.log('total_training_loss', sum(self.losses), on_epoch=True, prog_bar=True, logger=True)
# #             self.log('total_training_time', sum(self.total_training_times), on_epoch=True, prog_bar=True, logger=True)
# #             torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_epoch_{self.current_epoch}.ckpt")
            
        
# #     def on_train_end(self):
# #         # Save the final model

# #         torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_final.ckpt")
# #         self.log('final_total_training_loss', sum(self.losses), logger=True)
# #         self.log('final_total_training_time', sum(self.total_training_times), logger=True)




# class ModelLightningModule(pl.LightningModule):
#     def __init__(self, model, loss_fn, lr, steps_til_summary, model_name, validation_fn=None):
#         super(ModelLightningModule, self).__init__()
#         self.model = model
#         self.loss_fn = loss_fn
#         self.lr = lr
#         self.steps_til_summary = steps_til_summary
#         self.model_name = model_name
#         self.validation_fn = validation_fn
#         self.total_training_time = 0
#         self.total_loss = 0
#         self.core_iteration_time = None
#         self.train_loss_epoch = None

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         model_input, gt = batch
#         core_start_time = time.time()
#         model_output = self(model_input)
#         losses = self.loss_fn(model_output, gt)
#         self.train_loss_epoch = sum([loss.mean() for loss in losses.values()])
        
#         core_end_time = time.time()
#         self.core_iteration_time = core_end_time - core_start_time
#         self.total_training_time += self.core_iteration_time
#         self.total_loss += self.train_loss_epoch.item()

#         return self.train_loss_epoch

#     def on_train_epoch_end(self):
#         self.log(f'{self.model_name}/train_loss', self.train_loss_epoch, on_epoch=True, prog_bar=True, logger=True)
#         self.log(f'{self.model_name}/iteration_time', self.core_iteration_time, on_epoch=True, prog_bar=True, logger=True)



#         if self.current_epoch % self.steps_til_summary == 0:
#             self.log(f'{self.model_name}/total_training_loss', self.total_loss, on_epoch=True, prog_bar=True, logger=True)
#             self.log(f'{self.model_name}/total_training_time', self.total_training_time, on_epoch=True, prog_bar=True, logger=True)
#             # checkpoint_path = os.path.join(self.logger.log_dir, f"{self.model_name}_epoch_{self.current_epoch + 1}.ckpt")
#             # torch.save(self.model.state_dict(), checkpoint_path)

#     def on_train_end(self):

#         final_checkpoint_path = os.path.join(self.logger.log_dir, f"{self.model_name}_final.ckpt")
#         torch.save(self.model.state_dict(), final_checkpoint_path)
#         #torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_final.ckpt")
#         # self.log('final_total_training_loss', self.total_loss, logger=True)
#         # self.log('final_total_training_time', self.total_training_time, logger=True)
        
#     # Log the final total training loss
#         if isinstance(self.logger, TensorBoardLogger):
#             self.logger.experiment.add_scalar(f'{self.model_name}/final_total_training_loss', self.total_loss, self.current_epoch)
#         else:
#             self.logger.log_metrics(f'{self.model_name}/final_total_training_loss', self.total_loss)
#         # Log the final total training time
#         if isinstance(self.logger, TensorBoardLogger):
#             self.logger.experiment.add_scalar(f'{self.model_name}/final_total_training_time', self.total_training_time, self.current_epoch)
#         else:
#             self.logger.log_metrics(f'{self.model_name}/final_total_training_time', self.total_training_time)


#     def validation_step(self, batch, batch_idx):
#         pass


#     # def validation_step(self, batch, batch_idx):
#     #     model_input, gt = batch
#     #     model_output = self(model_input)
#     #     losses = self.loss_fn(model_output, gt)
#     #     val_loss = sum([loss.mean() for loss in losses.values()])
#     #     self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#     #     return val_loss

#     # def on_validation_epoch_end(self):
#     # # If you want to perform any operations at the end of each validation epoch
#     #     pass

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         return optimizer

# class LinearDecaySchedule():
#     def __init__(self, start_val, final_val, num_steps):
#         self.start_val = start_val
#         self.final_val = final_val
#         self.num_steps = num_steps

#     def __call__(self, iter):
#         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


import torch
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

class ModelLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, steps_til_summary, model_name, validation_fn=None):
        super(ModelLightningModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.steps_til_summary = steps_til_summary
        self.model_name = model_name
        self.validation_fn = validation_fn
        self.total_training_time = 0
        self.total_loss = 0
        self.core_iteration_time = None
        self.train_loss_epoch = None
        self.train_losses= []
        self.iteration_times = []
        self.total_training_times = []


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        model_input, gt = batch
        core_start_time = time.time()
        model_output = self(model_input)
        losses = self.loss_fn(model_output, gt)
        self.train_loss_epoch = sum([loss.mean() for loss in losses.values()])
        
        core_end_time = time.time()
        self.core_iteration_time = core_end_time - core_start_time
        self.total_training_time += self.core_iteration_time
        self.train_losses.append(self.train_loss_epoch.item())
        self.iteration_times.append(self.core_iteration_time)
        self.total_training_times.append(self.total_training_time)
        return self.train_loss_epoch


    def on_train_epoch_end(self):
            # Use consistent metric names across models, but tag with model name
        self.log('train_loss_epoch', self.train_loss_epoch, on_epoch=True, prog_bar=True, logger=True)
        #self.log(f'train_loss_epoch/{self.model_name}', self.train_loss_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('iteration_time', self.core_iteration_time, on_epoch=True, prog_bar=True, logger=True)
        #self.log(f'iteration_time/{self.model_name}', self.core_iteration_time, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        if self.current_epoch % self.steps_til_summary == 0:
            self.log('total_training_time', self.total_training_time, on_epoch=True, prog_bar=True, logger=True)
            #self.log(f'total_training_loss/{self.model_name}', self.total_loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)
            #self.log(f'total_training_time/{self.model_name}', self.total_training_time, on_step= False, on_epoch=True, prog_bar=True, logger=True)


    def on_train_end(self):
        
        checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_checkpoint_path = os.path.join(checkpoint_dir, f"{self.model_name}_final.ckpt")
        torch.save(self.model.state_dict(), final_checkpoint_path)

        final_epoch_training_loss_text = f'Model: {self.model_name}, Final Epoch Training Loss: {self.train_losses[-1]}'
        final_total_training_time_text = f'Model: {self.model_name}, Final Total Training Time: {self.total_training_time}'
        self.logger.experiment.add_text('Final Epoch Training Loss', final_epoch_training_loss_text, self.current_epoch)
        self.logger.experiment.add_text('Final Total Training Time', final_total_training_time_text, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
